import {
  InstancedMesh,
  InstancedBufferAttribute,
  ConeGeometry,
  MeshLambertMaterial,
  Color,
  Vector3,
  Quaternion,
  Matrix4
} from 'three';

import { colorStr2Hex, colorAlpha } from './color-utils';

const MAX_ARROWS_DEFAULT = 20000;
const _zAxis = new Vector3(0, 0, 1);
const _v1 = new Vector3();
const _v2 = new Vector3();
const _dir = new Vector3();
const _quat = new Quaternion();
const _scale = new Vector3();
const _matrix = new Matrix4();

/**
 * InstancedArrowRenderer - Renders directional arrows via a single InstancedMesh.
 * Arrows are not interactive (raycast disabled).
 */
export default class InstancedArrowRenderer {
  constructor(scene, maxArrows = MAX_ARROWS_DEFAULT) {
    this.scene = scene;
    this.maxArrows = maxArrows;
    this.count = 0;

    this.linkDataArray = new Array(maxArrows);
    this.colorArray = new Float32Array(maxArrows * 3);
    this.opacityArray = new Float32Array(maxArrows);

    this.mesh = null;
    this._resolution = 8;
    this._created = false;
  }

  init(resolution = 8) {
    if (this._created) this.dispose();
    this._resolution = resolution;

    // Unit cone: height=1, radius=0.25, pivot at base, aligned along +Z
    const geometry = new ConeGeometry(0.25, 1, resolution);
    geometry.translate(0, 0.5, 0);    // pivot at base
    geometry.rotateX(Math.PI / 2);    // align along Z

    const material = new MeshLambertMaterial({
      transparent: true,
      depthWrite: false,
      opacity: 1.0
    });

    // Per-instance opacity
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
      if (shader.vertexShader === origVert || shader.fragmentShader === origFrag) {
        console.warn('[InstancedArrowRenderer] Shader injection FAILED');
      }
    };
    material.customProgramCacheKey = function() { return 'instancedArrowRenderer'; };

    this.mesh = new InstancedMesh(geometry, material, this.maxArrows);
    this.mesh.count = 0;
    this.mesh.frustumCulled = false;

    // Built-in per-instance color
    this.mesh.instanceColor = new InstancedBufferAttribute(this.colorArray, 3);

    // Custom opacity attribute
    this.opacityAttr = new InstancedBufferAttribute(this.opacityArray, 1);
    geometry.setAttribute('instanceOpacity', this.opacityAttr);

    // Arrows are not interactive — disable raycasting
    this.mesh.raycast = function() {};
    this.mesh.renderOrder = 5; // draw between edges (0) and nodes (10)

    this.scene.add(this.mesh);
    this._created = true;
  }

  /**
   * Digest: sync arrow data from visible links that have arrow length > 0.
   */
  digest(links, accessors) {
    if (!this._created) return;

    const { arrowLength: arrowLengthAccessor, arrowColor: arrowColorAccessor,
            linkColor: linkColorAccessor, linkOpacity: globalLinkOpacity } = accessors;
    const tmpColor = new Color();

    this.count = Math.min(links.length, this.maxArrows);
    this.mesh.count = this.count;

    for (let i = 0; i < this.count; i++) {
      const link = links[i];
      this.linkDataArray[i] = link;

      const arrowColor = arrowColorAccessor(link) || linkColorAccessor(link) || '#f0f0f0';
      const hex = colorStr2Hex(arrowColor);
      tmpColor.set(hex);

      const i3 = i * 3;
      this.colorArray[i3] = tmpColor.r;
      this.colorArray[i3 + 1] = tmpColor.g;
      this.colorArray[i3 + 2] = tmpColor.b;

      this.opacityArray[i] = globalLinkOpacity * 3 * colorAlpha(arrowColor);
    }

    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
    if (this.opacityAttr) this.opacityAttr.needsUpdate = true;
  }

  /**
   * Per-tick position/rotation update for arrows.
   * Replicates updateArrows() logic from forcegraph-kapsule.
   */
  updatePositions(isD3Sim, accessors) {
    if (!this._created || this.count === 0) return;

    const { arrowLength: arrowLengthAccessor, arrowRelPos: arrowRelPosAccessor,
            nodeVal: nodeValAccessor, nodeRelSize } = accessors;

    for (let i = 0; i < this.count; i++) {
      const link = this.linkDataArray[i];
      if (!link) continue;

      const pos = isD3Sim ? link : null;
      if (!pos) continue;

      const start = isD3Sim ? pos.source : null;
      const end = isD3Sim ? pos.target : null;
      if (!start || !end || start.x === undefined || end.x === undefined) continue;

      const startR = Math.cbrt(Math.max(0, nodeValAccessor(start) || 1)) * nodeRelSize;
      const endR = Math.cbrt(Math.max(0, nodeValAccessor(end) || 1)) * nodeRelSize;

      const arrowLength = arrowLengthAccessor(link);
      const arrowRelPos = arrowRelPosAccessor(link);

      const getPosAlongLine = link.__curve
        ? t => link.__curve.getPoint(t)
        : t => ({
            x: start.x + (end.x - start.x) * t || 0,
            y: (start.y || 0) + ((end.y || 0) - (start.y || 0)) * t,
            z: (start.z || 0) + ((end.z || 0) - (start.z || 0)) * t
          });

      const lineLen = link.__curve
        ? link.__curve.getLength()
        : Math.sqrt(
            Math.pow((end.x || 0) - (start.x || 0), 2) +
            Math.pow((end.y || 0) - (start.y || 0), 2) +
            Math.pow((end.z || 0) - (start.z || 0), 2)
          );

      if (lineLen < 1e-6) continue;

      const posAlongLine = startR + arrowLength + (lineLen - startR - endR - arrowLength) * arrowRelPos;
      const arrowHead = getPosAlongLine(posAlongLine / lineLen);
      const arrowTail = getPosAlongLine((posAlongLine - arrowLength) / lineLen);

      _v1.set(arrowTail.x || 0, arrowTail.y || 0, arrowTail.z || 0);
      _v2.set(arrowHead.x || 0, arrowHead.y || 0, arrowHead.z || 0);

      _dir.subVectors(_v2, _v1);
      const len = _dir.length();
      if (len < 1e-6) continue;
      _dir.normalize();

      _quat.setFromUnitVectors(_zAxis, _dir);
      // Scale: uniform by arrowLength (cone is unit size)
      _scale.set(arrowLength, arrowLength, arrowLength);
      _matrix.compose(_v1, _quat, _scale);

      this.mesh.setMatrixAt(i, _matrix);
    }

    this.mesh.instanceMatrix.needsUpdate = true;
  }

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
