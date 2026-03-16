import {
  InstancedMesh,
  InstancedBufferAttribute,
  CylinderGeometry,
  MeshLambertMaterial,
  LineSegments,
  BufferGeometry,
  BufferAttribute,
  ShaderMaterial,
  Color,
  Box3,
  Sphere,
  Vector3,
  Quaternion,
  Matrix4
} from 'three';

import { colorStr2Hex, colorAlpha } from './color-utils';

const MAX_LINKS_DEFAULT = 20000;
const DEFAULT_CURVE_SEGMENTS = 12;
const _zAxis = new Vector3(0, 0, 1);
const _v1 = new Vector3();
const _v2 = new Vector3();
const _dir = new Vector3();
const _quat = new Quaternion();
const _scale = new Vector3();
const _matrix = new Matrix4();

// Reusable point pool for curve tessellation — avoids per-frame GC pressure
const _curvePointPool = [];
function getCurvePointsReuse(curve, segments) {
  while (_curvePointPool.length <= segments) _curvePointPool.push(new Vector3());
  for (let i = 0; i <= segments; i++) {
    curve.getPoint(i / segments, _curvePointPool[i]);
  }
  return _curvePointPool;
}

/**
 * InstancedLinkRenderer - Renders links via LineSegments (width=0)
 * or InstancedMesh cylinders (width>0). One draw call each.
 *
 * Supports both straight and curved links in the same draw call.
 * Curved links are tessellated into multiple segments.
 *
 * For per-vertex RGBA in LineSegments we use a custom ShaderMaterial.
 * For cylinders, we use MeshLambertMaterial + onBeforeCompile for per-instance opacity.
 */
export default class InstancedLinkRenderer {
  constructor(scene, maxLinks = MAX_LINKS_DEFAULT, maxCurveSegments = DEFAULT_CURVE_SEGMENTS) {
    this.scene = scene;
    this.maxLinks = maxLinks;
    this.maxCurveSegments = maxCurveSegments;
    this.count = 0; // logical link count

    // Instance data
    this.linkDataArray = new Array(maxLinks); // linkIndex → link data

    // --- Line mode buffers (width=0) ---
    // Worst case: each link is a curve with maxCurveSegments segment-pairs (2 verts each)
    const maxLineVerts = maxLinks * maxCurveSegments * 2;
    this.linePositions = new Float32Array(maxLineVerts * 3);
    this.lineColors = new Float32Array(maxLineVerts * 4);
    // Per-link vertex mapping for variable-length links
    this.linkVertexOffset = new Uint32Array(maxLinks); // link i → start vertex index
    this.linkVertexCount = new Uint16Array(maxLinks);  // link i → vertex count
    this.totalVertices = 0;

    // --- Cylinder mode buffers (width>0) ---
    // Worst case: each link is a curve with maxCurveSegments sub-cylinders
    const maxCylInstances = maxLinks * maxCurveSegments;
    this.cylColorArray = new Float32Array(maxCylInstances * 3);
    this.cylOpacityArray = new Float32Array(maxCylInstances);
    // Sub-instance mapping for cylinders
    this.instanceToLinkIndex = new Uint32Array(maxCylInstances); // sub-instance → link index
    this.linkInstanceOffset = new Uint32Array(maxLinks); // link index → first sub-instance
    this.linkInstanceCount = new Uint16Array(maxLinks);  // link index → sub-instance count
    this.totalInstances = 0;

    this.lineSegments = null;
    this.cylinderMesh = null;
    this._mode = null; // 'line' or 'cylinder'
    this._resolution = 6;
    this._created = false;
  }

  /**
   * Initialize renderer in line or cylinder mode.
   */
  init(useCylinder = false, resolution = 6) {
    if (this._created) this.dispose();
    this._resolution = resolution;
    this._mode = useCylinder ? 'cylinder' : 'line';

    if (useCylinder) {
      this._initCylinder(resolution);
    } else {
      this._initLineSegments();
    }

    this._created = true;
  }

  _initLineSegments() {
    const geometry = new BufferGeometry();
    const posAttr = new BufferAttribute(this.linePositions, 3);
    posAttr.setUsage(35048); // DynamicDrawUsage
    geometry.setAttribute('position', posAttr);

    const colorAttr = new BufferAttribute(this.lineColors, 4);
    colorAttr.setUsage(35048);
    geometry.setAttribute('aColor', colorAttr);

    // Custom shader for per-vertex RGBA
    const material = new ShaderMaterial({
      vertexShader: `
        attribute vec4 aColor;
        varying vec4 vColor;
        void main() {
          vColor = aColor;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec4 vColor;
        void main() {
          if (vColor.a < 0.01) discard;
          gl_FragColor = vColor;
        }
      `,
      transparent: true,
      depthWrite: false,
    });

    this.lineSegments = new LineSegments(geometry, material);
    this.lineSegments.renderOrder = 10; // render links after opaque passes
    this.lineSegments.frustumCulled = false;

    // Tag for interaction picking
    this.lineSegments.__graphObjType = 'link';
    this.lineSegments.__isInstancedRenderer = true;
    this.lineSegments.__instancedRenderer = this;

    // Override raycast to inject instanceId (= link index) from vertex index
    const originalRaycast = this.lineSegments.raycast.bind(this.lineSegments);
    const self = this;
    this.lineSegments.raycast = function(raycaster, intersects) {
      const before = intersects.length;
      originalRaycast(raycaster, intersects);
      // Tag new intersections with link index (binary search for variable-length links)
      for (let j = before; j < intersects.length; j++) {
        intersects[j].instanceId = self._vertexToLinkIndex(intersects[j].index);
      }
    };

    // Large bounding sphere to prevent stale cache misses
    geometry.boundingSphere = new Sphere(new Vector3(0, 0, 0), 1e10);

    this.scene.add(this.lineSegments);
  }

  _initCylinder(resolution) {
    // Unit cylinder: radius=1, height=1, pivot at bottom, aligned along +Z
    const geometry = new CylinderGeometry(1, 1, 1, resolution, 1, false);
    geometry.translate(0, 0.5, 0);      // pivot at base
    geometry.rotateX(Math.PI / 2);      // align along Z axis

    const material = new MeshLambertMaterial({
      transparent: true,
      depthWrite: false,
      opacity: 1.0
    });

    // Inject per-instance opacity
    material.onBeforeCompile = (shader) => {
      const origVert = shader.vertexShader;
      shader.vertexShader = shader.vertexShader.replace(
        'void main() {',
        'attribute float instanceOpacity;\nvarying float vInstanceOpacity;\nvoid main() {\n  vInstanceOpacity = instanceOpacity;'
      );
      shader.fragmentShader = shader.fragmentShader.replace(
        'void main() {',
        'varying float vInstanceOpacity;\nvoid main() {'
      );
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
        console.warn('[InstancedLinkRenderer:cyl] Vertex shader injection FAILED');
      }
      if (shader.fragmentShader === origFrag) {
        console.warn('[InstancedLinkRenderer:cyl] Fragment shader opacity injection FAILED');
      }
    };
    material.customProgramCacheKey = function() { return 'instancedLinkCylRenderer'; };

    // Allocate for worst case: each link could have maxCurveSegments sub-cylinders
    const maxInstances = this.maxLinks * this.maxCurveSegments;
    this.cylinderMesh = new InstancedMesh(geometry, material, maxInstances);
    this.cylinderMesh.count = 0;
    this.cylinderMesh.frustumCulled = false;
    this.cylinderMesh.renderOrder = 10; // match LineSegments render order

    // Enable built-in per-instance color
    this.cylinderMesh.instanceColor = new InstancedBufferAttribute(this.cylColorArray, 3);

    // Custom opacity attribute
    this.cylOpacityAttr = new InstancedBufferAttribute(this.cylOpacityArray, 1);
    geometry.setAttribute('instanceOpacity', this.cylOpacityAttr);

    // Tag for picking
    this.cylinderMesh.__graphObjType = 'link';
    this.cylinderMesh.__isInstancedRenderer = true;
    this.cylinderMesh.__instancedRenderer = this;

    // Large bounding sphere to prevent stale cache misses
    this.cylinderMesh.boundingSphere = new Sphere(new Vector3(0, 0, 0), 1e10);

    this.scene.add(this.cylinderMesh);
  }

  /**
   * Digest: sync link data into instance arrays.
   */
  digest(links, accessors) {
    if (!this._created) return;

    const { color: colorAccessor, opacity: globalOpacity, width: widthAccessor } = accessors;
    const tmpColor = new Color();

    this.count = Math.min(links.length, this.maxLinks);

    if (this._mode === 'line') {
      this._digestLine(links, colorAccessor, globalOpacity, tmpColor);
    } else {
      this._digestCylinder(links, colorAccessor, globalOpacity, tmpColor);
    }

    // Clear stale entries
    for (let i = this.count; i < this.linkDataArray.length; i++) {
      if (this.linkDataArray[i] === undefined) break;
      this.linkDataArray[i] = undefined;
    }

    // Write initial positions
    this.updatePositions(true);

    // Clear individual link Three.js object refs
    for (let i = 0; i < this.count; i++) {
      links[i].__lineObj = undefined;
    }
  }

  _digestLine(links, colorAccessor, globalOpacity, tmpColor) {
    const segs = this.maxCurveSegments;
    let vertexOffset = 0;

    for (let i = 0; i < this.count; i++) {
      const link = links[i];
      this.linkDataArray[i] = link;

      // Determine vertex count: curved links get 2*segs vertices, straight get 2
      const isCurved = !!link.__curve || !!link.curvature;
      const vertCount = isCurved ? segs * 2 : 2;

      this.linkVertexOffset[i] = vertexOffset;
      this.linkVertexCount[i] = vertCount;

      const colorStr = colorAccessor(link);
      const hex = colorStr2Hex(colorStr || '#f0f0f0');
      tmpColor.set(hex);
      const alpha = globalOpacity * colorAlpha(colorStr);

      // Set color for all vertices of this link
      for (let v = 0; v < vertCount; v++) {
        const ci = (vertexOffset + v) * 4;
        this.lineColors[ci]     = tmpColor.r;
        this.lineColors[ci + 1] = tmpColor.g;
        this.lineColors[ci + 2] = tmpColor.b;
        this.lineColors[ci + 3] = alpha;
      }

      vertexOffset += vertCount;
    }

    this.totalVertices = vertexOffset;
    this.lineSegments.geometry.setDrawRange(0, this.totalVertices);
    this.lineSegments.geometry.attributes.aColor.needsUpdate = true;
  }

  _digestCylinder(links, colorAccessor, globalOpacity, tmpColor) {
    const segs = this.maxCurveSegments;
    let instanceOffset = 0;

    for (let i = 0; i < this.count; i++) {
      const link = links[i];
      this.linkDataArray[i] = link;

      const isCurved = !!link.__curve || !!link.curvature;
      const subCount = isCurved ? segs : 1;

      this.linkInstanceOffset[i] = instanceOffset;
      this.linkInstanceCount[i] = subCount;

      const colorStr = colorAccessor(link);
      const hex = colorStr2Hex(colorStr || '#f0f0f0');
      tmpColor.set(hex);
      const alpha = globalOpacity * colorAlpha(colorStr);

      for (let s = 0; s < subCount; s++) {
        const idx = instanceOffset + s;
        this.instanceToLinkIndex[idx] = i;
        const i3 = idx * 3;
        this.cylColorArray[i3]     = tmpColor.r;
        this.cylColorArray[i3 + 1] = tmpColor.g;
        this.cylColorArray[i3 + 2] = tmpColor.b;
        this.cylOpacityArray[idx] = alpha;
      }

      instanceOffset += subCount;
    }

    this.totalInstances = instanceOffset;
    this.cylinderMesh.count = this.totalInstances;

    if (this.cylinderMesh.instanceColor) this.cylinderMesh.instanceColor.needsUpdate = true;
    if (this.cylOpacityAttr) this.cylOpacityAttr.needsUpdate = true;
  }

  /**
   * Per-tick position update.
   */
  updatePositions(isD3Sim, linkWidthVal) {
    if (!this._created || this.count === 0) return;

    if (this._mode === 'line') {
      this._updateLinePositions(isD3Sim);
    } else {
      this._updateCylinderPositions(isD3Sim, linkWidthVal);
    }
  }

  _updateLinePositions(isD3Sim) {
    const pos = this.linePositions;
    const segs = this.maxCurveSegments;

    for (let i = 0; i < this.count; i++) {
      const link = this.linkDataArray[i];
      if (!link) continue;

      const src = isD3Sim ? link.source : null;
      const tgt = isD3Sim ? link.target : null;
      if (!src || !tgt || src.x === undefined || tgt.x === undefined) continue;

      const offset = this.linkVertexOffset[i];
      const curve = link.__curve;

      if (!curve) {
        // Straight line: 2 vertices (1 segment-pair)
        const o3 = offset * 3;
        pos[o3]     = src.x;
        pos[o3 + 1] = src.y || 0;
        pos[o3 + 2] = src.z || 0;
        pos[o3 + 3] = tgt.x;
        pos[o3 + 4] = tgt.y || 0;
        pos[o3 + 5] = tgt.z || 0;
      } else {
        // Curved line: tessellate into segment-pairs for LineSegments format
        const points = getCurvePointsReuse(curve, segs);
        let vi = offset * 3;
        for (let s = 0; s < segs; s++) {
          const p0 = points[s];
          const p1 = points[s + 1];
          pos[vi++] = p0.x;
          pos[vi++] = p0.y;
          pos[vi++] = p0.z;
          pos[vi++] = p1.x;
          pos[vi++] = p1.y;
          pos[vi++] = p1.z;
        }
      }
    }

    this.lineSegments.geometry.attributes.position.needsUpdate = true;
  }

  _updateCylinderPositions(isD3Sim, linkWidthVal) {
    const r = (linkWidthVal || 1) / 2;
    const segs = this.maxCurveSegments;

    for (let i = 0; i < this.count; i++) {
      const link = this.linkDataArray[i];
      const instOffset = this.linkInstanceOffset[i];
      const instCount = this.linkInstanceCount[i];

      if (!link) {
        // Hide all sub-instances
        for (let s = 0; s < instCount; s++) {
          _scale.set(0, 0, 0);
          _matrix.compose(_v1, _quat, _scale);
          this.cylinderMesh.setMatrixAt(instOffset + s, _matrix);
        }
        continue;
      }

      const src = isD3Sim ? link.source : null;
      const tgt = isD3Sim ? link.target : null;
      if (!src || !tgt || src.x === undefined || tgt.x === undefined) {
        for (let s = 0; s < instCount; s++) {
          _scale.set(0, 0, 0);
          _matrix.compose(_v1, _quat, _scale);
          this.cylinderMesh.setMatrixAt(instOffset + s, _matrix);
        }
        continue;
      }

      const curve = link.__curve;

      if (!curve) {
        // Straight: single cylinder
        _v1.set(src.x, src.y || 0, src.z || 0);
        _v2.set(tgt.x, tgt.y || 0, tgt.z || 0);
        const dist = _v1.distanceTo(_v2);
        if (dist < 1e-6) {
          _scale.set(0, 0, 0);
          _matrix.compose(_v1, _quat, _scale);
        } else {
          _dir.subVectors(_v2, _v1).normalize();
          _quat.setFromUnitVectors(_zAxis, _dir);
          _scale.set(r, r, dist);
          _matrix.compose(_v1, _quat, _scale);
        }
        this.cylinderMesh.setMatrixAt(instOffset, _matrix);
      } else {
        // Curved: N sub-cylinders along curve
        const points = getCurvePointsReuse(curve, segs);
        for (let s = 0; s < segs; s++) {
          _v1.set(points[s].x, points[s].y, points[s].z);
          _v2.set(points[s + 1].x, points[s + 1].y, points[s + 1].z);
          const dist = _v1.distanceTo(_v2);
          if (dist < 1e-6) {
            _scale.set(0, 0, 0);
            _matrix.compose(_v1, _quat, _scale);
          } else {
            _dir.subVectors(_v2, _v1).normalize();
            _quat.setFromUnitVectors(_zAxis, _dir);
            _scale.set(r, r, dist);
            _matrix.compose(_v1, _quat, _scale);
          }
          this.cylinderMesh.setMatrixAt(instOffset + s, _matrix);
        }
      }
    }

    this.cylinderMesh.instanceMatrix.needsUpdate = true;
  }

  /**
   * Binary search: vertex index → link index (for line mode raycast).
   * linkVertexOffset is monotonically increasing.
   */
  _vertexToLinkIndex(vertIdx) {
    let lo = 0, hi = this.count - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const start = this.linkVertexOffset[mid];
      const end = start + this.linkVertexCount[mid];
      if (vertIdx < start) hi = mid - 1;
      else if (vertIdx >= end) lo = mid + 1;
      else return mid;
    }
    return -1;
  }

  /**
   * Look up link data from instanceId.
   */
  getDataByInstanceId(instanceId) {
    if (this._mode === 'cylinder' && this.instanceToLinkIndex) {
      if (instanceId >= 0 && instanceId < this.totalInstances) {
        const linkIdx = this.instanceToLinkIndex[instanceId];
        return this.linkDataArray[linkIdx] || null;
      }
      return null;
    }
    // Line mode: instanceId is already the link index (set by raycast override)
    if (instanceId >= 0 && instanceId < this.count) {
      return this.linkDataArray[instanceId];
    }
    return null;
  }

  /**
   * Check if a Three.js object belongs to this renderer.
   */
  isOurMesh(obj) {
    return obj === this.lineSegments || obj === this.cylinderMesh;
  }

  /**
   * Get the active Three.js object (for adding to scene objects, etc.)
   */
  getObject() {
    return this._mode === 'line' ? this.lineSegments : this.cylinderMesh;
  }

  /**
   * Compute bounding box from link positions.
   */
  computeBBox() {
    if (this.count === 0) return null;

    const box = new Box3();
    const _v = new Vector3();

    for (let i = 0; i < this.count; i++) {
      const link = this.linkDataArray[i];
      if (!link) continue;

      const src = link.source;
      const tgt = link.target;
      if (!src || !tgt) continue;

      if (src.x !== undefined) {
        _v.set(src.x, src.y || 0, src.z || 0);
        box.expandByPoint(_v);
      }
      if (tgt.x !== undefined) {
        _v.set(tgt.x, tgt.y || 0, tgt.z || 0);
        box.expandByPoint(_v);
      }
    }

    return box.isEmpty() ? null : box;
  }

  dispose() {
    if (this.lineSegments) {
      this.scene.remove(this.lineSegments);
      this.lineSegments.geometry.dispose();
      this.lineSegments.material.dispose();
      this.lineSegments = null;
    }
    if (this.cylinderMesh) {
      this.scene.remove(this.cylinderMesh);
      this.cylinderMesh.geometry.dispose();
      this.cylinderMesh.material.dispose();
      this.cylinderMesh = null;
    }
    this._created = false;
    this._mode = null;
    this.count = 0;
    this.totalVertices = 0;
    this.totalInstances = 0;
  }
}
