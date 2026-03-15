import {
  InstancedMesh,
  InstancedBufferAttribute,
  SphereGeometry,
  MeshLambertMaterial,
  Color,
  Box3,
  Sphere,
  Vector3
} from 'three';

import { colorStr2Hex, colorAlpha } from './color-utils';

const MAX_NODES_DEFAULT = 100000;
const _camPos = new Vector3();

/**
 * InstancedNodeRenderer - Renders all default (non-custom) nodes via a single InstancedMesh.
 *
 * One draw call for all spherical nodes. Per-instance color via built-in instanceColor,
 * per-instance opacity via onBeforeCompile injection into MeshLambertMaterial.
 *
 * Uses per-frame back-to-front sorting so that depthWrite:true works correctly
 * with transparent instances (nodes further from camera render first).
 */
export default class InstancedNodeRenderer {
  constructor(scene, maxNodes = MAX_NODES_DEFAULT) {
    this.scene = scene;
    this.maxNodes = maxNodes;
    this.count = 0;

    // Instance data arrays (ORIGINAL order — never rearranged)
    this.nodeDataArray = new Array(maxNodes); // origIdx → node data reference
    this.radii = new Float32Array(maxNodes);

    // Source arrays in original order (written by digest)
    this._srcColorArray = new Float32Array(maxNodes * 3);
    this._srcOpacityArray = new Float32Array(maxNodes);

    // GPU arrays (written in sorted order each frame)
    this.colorArray = new Float32Array(maxNodes * 3);
    this.opacityArray = new Float32Array(maxNodes);

    // Sort indirection: _sortIndices[sortedPos] = origIdx
    this._sortIndices = new Uint16Array(maxNodes);
    this._distances = new Float32Array(maxNodes);

    // Camera cache — skip sort when camera hasn't moved
    this._lastCamX = NaN;
    this._lastCamY = NaN;
    this._lastCamZ = NaN;
    this._positionsDirty = true; // force sort on first frame and after digest
    this._colorsSorted = false; // true when colorArray is in camera-distance order

    // Defaults
    this._srcOpacityArray.fill(0.75);
    this.opacityArray.fill(0.75);

    this.mesh = null;
    this._resolution = 8;
    this._created = false;
  }

  /**
   * Create the InstancedMesh. Call once after scene is ready.
   */
  init(resolution = 8) {
    if (this._created) this.dispose();
    this._resolution = resolution;

    const geometry = new SphereGeometry(1, resolution, resolution);

    const material = new MeshLambertMaterial({
      transparent: true,
      depthWrite: true, // Sorted back-to-front each frame so depth writes are correct
      opacity: 1.0 // global opacity handled per-instance
    });

    // Inject per-instance opacity via onBeforeCompile
    material.onBeforeCompile = (shader) => {
      // Vertex: declare attribute + varying
      const origVert = shader.vertexShader;
      shader.vertexShader = shader.vertexShader.replace(
        'void main() {',
        'attribute float instanceOpacity;\nvarying float vInstanceOpacity;\nvoid main() {\n  vInstanceOpacity = instanceOpacity;'
      );

      // Fragment: declare varying
      shader.fragmentShader = shader.fragmentShader.replace(
        'void main() {',
        'varying float vInstanceOpacity;\nvoid main() {'
      );

      // Fragment: replace #include <opaque_fragment> with custom version that applies per-instance opacity.
      // NOTE: onBeforeCompile receives the shader BEFORE #include directives are resolved,
      // so we must target the #include directive, not the resolved gl_FragColor line.
      const origFrag = shader.fragmentShader;
      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <opaque_fragment>',
        [
          '#ifdef OPAQUE',
          '  diffuseColor.a = 1.0;',
          '#endif',
          '#ifdef USE_TRANSMISSION',
          '  diffuseColor.a *= material.transmissionAlpha;',
          '#endif',
          'gl_FragColor = vec4( outgoingLight, diffuseColor.a * vInstanceOpacity );',
          'if (gl_FragColor.a < 0.01) discard;'
        ].join('\n')
      );

      if (shader.vertexShader === origVert) {
        console.warn('[InstancedNodeRenderer] Vertex shader injection FAILED');
      }
      if (shader.fragmentShader === origFrag) {
        console.warn('[InstancedNodeRenderer] Fragment shader opacity injection FAILED');
        console.warn('[InstancedNodeRenderer] Fragment shader (first 500 chars):', shader.fragmentShader.substring(0, 500));
      }
    };
    // Unique cache key to prevent shader cache collisions with standard MeshLambertMaterial
    material.customProgramCacheKey = function() { return 'instancedNodeRenderer'; };

    this.mesh = new InstancedMesh(geometry, material, this.maxNodes);
    this.mesh.count = 0;
    this.mesh.frustumCulled = false;

    // Enable built-in per-instance color
    const colorAttr = new InstancedBufferAttribute(this.colorArray, 3);
    this.mesh.instanceColor = colorAttr;

    // Add custom per-instance opacity attribute
    this.opacityAttr = new InstancedBufferAttribute(this.opacityArray, 1);
    geometry.setAttribute('instanceOpacity', this.opacityAttr);

    // Tag mesh for interaction picking
    this.mesh.__graphObjType = 'node';
    this.mesh.__isInstancedRenderer = true;
    this.mesh.__instancedRenderer = this;

    // Set large bounding sphere so raycasting always reaches per-instance tests.
    this.mesh.boundingSphere = new Sphere(new Vector3(0, 0, 0), 1e10);

    this.scene.add(this.mesh);
    this._created = true;
  }

  /**
   * Digest: sync node data into instance arrays.
   * Called when data or visual properties change (not per-tick).
   */
  digest(nodes, accessors) {
    if (!this._created) return;

    const { val: valAccessor, color: colorAccessor, opacity: globalOpacity, relSize } = accessors;
    const tmpColor = new Color();

    this.count = Math.min(nodes.length, this.maxNodes);
    this.mesh.count = this.count;

    for (let i = 0; i < this.count; i++) {
      const node = nodes[i];
      this.nodeDataArray[i] = node;

      // Radius: cbrt(val) * relSize (matches forcegraph-kapsule formula)
      const val = valAccessor(node) || 1;
      const radius = Math.cbrt(val) * relSize;
      this.radii[i] = radius;

      // Color — write to source arrays (original order)
      const colorStr = colorAccessor(node);
      const hex = colorStr2Hex(colorStr || '#ffffaa');
      tmpColor.set(hex);
      const i3 = i * 3;
      this._srcColorArray[i3] = tmpColor.r;
      this._srcColorArray[i3 + 1] = tmpColor.g;
      this._srcColorArray[i3 + 2] = tmpColor.b;

      // Opacity — write to source array (original order)
      this._srcOpacityArray[i] = globalOpacity * colorAlpha(colorStr);

      // Initialize sort indices to identity
      this._sortIndices[i] = i;
    }

    // Clear stale entries beyond count
    for (let i = this.count; i < this.nodeDataArray.length; i++) {
      if (this.nodeDataArray[i] === undefined) break;
      this.nodeDataArray[i] = undefined;
    }

    // Copy source → GPU arrays (unsorted initially, will be sorted on first frame)
    this.colorArray.set(this._srcColorArray.subarray(0, this.count * 3));
    this.opacityArray.set(this._srcOpacityArray.subarray(0, this.count));

    // Mark attributes dirty
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
    if (this.opacityAttr) this.opacityAttr.needsUpdate = true;

    // Write initial positions from node data
    this._positionsDirty = true; // force re-sort on next frame
    this.updatePositions(true);

    // Clear individual mesh refs (instanced nodes don't have one)
    for (let i = 0; i < this.count; i++) {
      nodes[i].__threeObj = undefined;
    }
  }

  /**
   * Per-tick position update WITHOUT sort (used as fallback / initial).
   * Writes instanceMatrix from node.x/y/z.
   */
  updatePositions(isD3Sim) {
    if (!this._created || this.count === 0) return;

    // When switching from sorted → unsorted order, restore original color/opacity
    if (this._colorsSorted) {
      this.colorArray.set(this._srcColorArray.subarray(0, this.count * 3));
      this.opacityArray.set(this._srcOpacityArray.subarray(0, this.count));
      if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
      if (this.opacityAttr) this.opacityAttr.needsUpdate = true;
      this._colorsSorted = false;
    }

    const arr = this.mesh.instanceMatrix.array;

    for (let i = 0; i < this.count; i++) {
      const node = this.nodeDataArray[i];
      if (!node) continue;

      const pos = isD3Sim ? node : null;
      if (!pos) continue;

      const r = this.radii[i];
      const offset = i * 16;

      // Column-major 4x4: scale (r,r,r) + translate (x,y,z)
      arr[offset]     = r;
      arr[offset + 1] = 0;
      arr[offset + 2] = 0;
      arr[offset + 3] = 0;
      arr[offset + 4] = 0;
      arr[offset + 5] = r;
      arr[offset + 6] = 0;
      arr[offset + 7] = 0;
      arr[offset + 8] = 0;
      arr[offset + 9] = 0;
      arr[offset + 10] = r;
      arr[offset + 11] = 0;
      arr[offset + 12] = pos.x || 0;
      arr[offset + 13] = pos.y || 0;
      arr[offset + 14] = pos.z || 0;
      arr[offset + 15] = 1;
    }

    this.mesh.instanceMatrix.needsUpdate = true;
  }

  /**
   * Per-tick sorted position update. Sorts instances back-to-front by camera distance
   * then writes instanceMatrix, instanceColor, and instanceOpacity in sorted order.
   * This ensures correct transparency with depthWrite:true.
   */
  sortAndUpdatePositions(camera, isD3Sim) {
    if (!this._created || this.count === 0) return;
    if (!camera) {
      this.updatePositions(isD3Sim);
      return;
    }

    const count = this.count;

    // Check if camera moved (skip expensive sort+copy when idle)
    camera.getWorldPosition(_camPos);
    const cx = _camPos.x;
    const cy = _camPos.y;
    const cz = _camPos.z;

    const dx = cx - this._lastCamX;
    const dy = cy - this._lastCamY;
    const dz = cz - this._lastCamZ;
    const camMoved = this._positionsDirty || (dx * dx + dy * dy + dz * dz) > 0.01;

    if (!camMoved) return; // nothing changed — GPU buffers are still valid

    this._lastCamX = cx;
    this._lastCamY = cy;
    this._lastCamZ = cz;
    this._positionsDirty = false;

    // Compute squared distances from camera to each node (original order)
    const distances = this._distances;
    for (let i = 0; i < count; i++) {
      const node = this.nodeDataArray[i];
      if (!node) { distances[i] = 0; continue; }
      const ndx = (node.x || 0) - cx;
      const ndy = (node.y || 0) - cy;
      const ndz = (node.z || 0) - cz;
      distances[i] = ndx * ndx + ndy * ndy + ndz * ndz;
    }

    // Build sort indices and sort back-to-front (farthest first)
    const sortIndices = this._sortIndices;
    for (let i = 0; i < count; i++) sortIndices[i] = i;
    sortIndices.subarray(0, count).sort((a, b) => distances[b] - distances[a]);

    // Write GPU arrays in sorted order
    const matArr = this.mesh.instanceMatrix.array;
    const colArr = this.colorArray;
    const opaArr = this.opacityArray;
    const srcCol = this._srcColorArray;
    const srcOpa = this._srcOpacityArray;

    for (let sortedPos = 0; sortedPos < count; sortedPos++) {
      const origIdx = sortIndices[sortedPos];
      const node = this.nodeDataArray[origIdx];

      const r = this.radii[origIdx];
      const offset = sortedPos * 16;

      if (!node) {
        for (let k = 0; k < 16; k++) matArr[offset + k] = 0;
        continue;
      }

      const px = isD3Sim ? (node.x || 0) : 0;
      const py = isD3Sim ? (node.y || 0) : 0;
      const pz = isD3Sim ? (node.z || 0) : 0;

      matArr[offset]     = r;
      matArr[offset + 1] = 0;
      matArr[offset + 2] = 0;
      matArr[offset + 3] = 0;
      matArr[offset + 4] = 0;
      matArr[offset + 5] = r;
      matArr[offset + 6] = 0;
      matArr[offset + 7] = 0;
      matArr[offset + 8] = 0;
      matArr[offset + 9] = 0;
      matArr[offset + 10] = r;
      matArr[offset + 11] = 0;
      matArr[offset + 12] = px;
      matArr[offset + 13] = py;
      matArr[offset + 14] = pz;
      matArr[offset + 15] = 1;

      const srcI3 = origIdx * 3;
      const dstI3 = sortedPos * 3;
      colArr[dstI3]     = srcCol[srcI3];
      colArr[dstI3 + 1] = srcCol[srcI3 + 1];
      colArr[dstI3 + 2] = srcCol[srcI3 + 2];

      opaArr[sortedPos] = srcOpa[origIdx];
    }

    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
    if (this.opacityAttr) this.opacityAttr.needsUpdate = true;
    this._colorsSorted = true;
  }

  /**
   * Look up node data from raycast instanceId.
   * After sorting, instanceId maps to a sorted position — use _sortIndices to resolve.
   */
  getDataByInstanceId(instanceId) {
    if (instanceId >= 0 && instanceId < this.count) {
      const origIdx = this._sortIndices[instanceId];
      return this.nodeDataArray[origIdx];
    }
    return null;
  }

  /**
   * Get position for a given instanceId (for drag proxy, etc.)
   */
  getPositionByInstanceId(instanceId) {
    const node = this.getDataByInstanceId(instanceId);
    if (node) {
      return new Vector3(node.x || 0, node.y || 0, node.z || 0);
    }
    return new Vector3();
  }

  /**
   * Check if a Three.js object is our instanced mesh.
   */
  isOurMesh(obj) {
    return obj === this.mesh;
  }

  /**
   * Compute bounding box from instance positions (for getGraphBbox).
   */
  computeBBox(nodeFilter) {
    if (this.count === 0) return null;

    const box = new Box3();
    const _v = new Vector3();

    for (let i = 0; i < this.count; i++) {
      const node = this.nodeDataArray[i];
      if (!node) continue;
      if (nodeFilter && !nodeFilter(node)) continue;

      const r = this.radii[i];
      _v.set(node.x || 0, node.y || 0, node.z || 0);
      box.expandByPoint(_v.clone().addScalar(r));
      box.expandByPoint(_v.clone().subScalar(r));
    }

    return box.isEmpty() ? null : box;
  }

  /**
   * Update resolution (sphere segment count). Requires reinit.
   */
  setResolution(resolution) {
    if (resolution !== this._resolution) {
      const wasCreated = this._created;
      if (wasCreated) {
        const savedNodes = this.nodeDataArray.slice(0, this.count);
        this.init(resolution);
        // Re-digest would be needed from the caller
      }
    }
  }

  /**
   * Dispose all GPU resources.
   */
  dispose() {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh.material.dispose();
      this.mesh = null;
    }
    this._created = false;
    this.count = 0;
  }
}
