import {
  Group,
  Mesh,
  MeshLambertMaterial,
  Color,
  BufferGeometry,
  BufferAttribute,
  Matrix4,
  Vector3,
  SphereGeometry,
  CylinderGeometry,
  TubeGeometry,
  ConeGeometry,
  Line,
  LineBasicMaterial,
  QuadraticBezierCurve3,
  CubicBezierCurve3,
  Box3
} from 'three';

const three = window.THREE
  ? window.THREE // Prefer consumption from global THREE, if exists
  : {
    Group,
    Mesh,
    MeshLambertMaterial,
    Color,
    BufferGeometry,
    BufferAttribute,
    Matrix4,
    Vector3,
    SphereGeometry,
    CylinderGeometry,
    TubeGeometry,
    ConeGeometry,
    Line,
    LineBasicMaterial,
    QuadraticBezierCurve3,
    CubicBezierCurve3,
    Box3
  };

import {
  forceSimulation as d3ForceSimulation,
  forceLink as d3ForceLink,
  forceManyBody as d3ForceManyBody,
  forceCenter as d3ForceCenter,
  forceRadial as d3ForceRadial
} from 'd3-force-3d';

import graph from 'ngraph.graph';
import forcelayout from 'ngraph.forcelayout';
const ngraph = { graph, forcelayout };

import Kapsule from 'kapsule';
import accessorFn from 'accessor-fn';

import { min as d3Min, max as d3Max } from 'd3-array';

// Self-loop curve resolution (lower than the default 30 for regular curves — fewer vertices, cheaper TubeGeometry)
const SELF_LOOP_CURVE_RESOLUTION = 12;

// Reusable Vector3 pool for self-loop billboard calculations (avoids per-frame allocation)
const _slCamDir = new three.Vector3();
const _slAxis1 = new three.Vector3();
const _slAxis2 = new three.Vector3();
const _slCp1 = new three.Vector3();
const _slCp2 = new three.Vector3();
const _slStart = new three.Vector3();
const _slLastCamDir = new three.Vector3(NaN, NaN, NaN); // sentinel: force first update

import ThreeDigest from './utils/three-digest';
import { emptyObject } from './utils/three-gc';
import { autoColorObjects, colorStr2Hex, colorAlpha } from './utils/color-utils';
import getDagDepths from './utils/dagDepths';
import InstancedNodeRenderer from './utils/InstancedNodeRenderer';
import InstancedLinkRenderer from './utils/InstancedLinkRenderer';
import InstancedArrowRenderer from './utils/InstancedArrowRenderer';

//

const DAG_LEVEL_NODE_RATIO = 2;

// support multiple method names for backwards threejs compatibility
const setAttributeFn = new three.BufferGeometry().setAttribute ? 'setAttribute' : 'addAttribute';
const applyMatrix4Fn = new three.BufferGeometry().applyMatrix4 ? 'applyMatrix4' : 'applyMatrix';

// Billboard self-loop update: recalculates self-loop curves to face the camera.
// Called every frame from tickFrame (outside layoutTick) so loops stay camera-facing
// even after simulation stops. Uses module-level Vector3 pool to avoid GC pressure.
// Only iterates state._selfLoopLinks (populated by calcLinkCurve) — O(selfLoops), not O(allLinks).
function updateSelfLoopBillboards(state, camera, isD3Sim) {
  const selfLoops = state._selfLoopLinks;
  if (!selfLoops || selfLoops.length === 0) return;

  camera.getWorldDirection(_slCamDir);

  // Skip entirely if camera direction hasn't changed — avoids expensive
  // geometry rebuilds (TubeGeometry) and Vector3 allocations (getPoints) every idle frame.
  if (_slCamDir.equals(_slLastCamDir)) return;

  const linkWidthAccessor = accessorFn(state.linkWidth);

  // Build camera-facing plane axes (perpendicular to camera look direction)
  if (Math.abs(_slCamDir.y) < 0.9) {
    _slAxis1.set(0, 1, 0).cross(_slCamDir).normalize();
  } else {
    _slAxis1.set(1, 0, 0).cross(_slCamDir).normalize();
  }
  _slAxis2.crossVectors(_slCamDir, _slAxis1).normalize();

  let anySkipped = false;

  for (let i = 0, len = selfLoops.length; i < len; i++) {
    const link = selfLoops[i];

    // Skip self-loops whose visual is hidden (e.g. hidden during camera movement)
    const lineObj = state.linkDataMapper ? state.linkDataMapper.getObj(link) : null;
    if (!lineObj || !lineObj.visible) { anySkipped = true; continue; }

    const pos = isD3Sim ? link : state.layout.getLinkPosition(state.layout.graph.getLink(link.source, link.target).id);
    const start = pos[isD3Sim ? 'source' : 'from'];
    if (!start || !start.hasOwnProperty('x')) continue;

    // Use pre-cached cos/sin (constant per link, computed once in calcLinkCurve)
    _slStart.set(start.x, start.y || 0, start.z || 0);

    const d = link.__selfLoopD;

    // Compute control points in camera-facing plane
    _slCp1.set(0, 0, 0)
      .addScaledVector(_slAxis1, d * link.__selfLoopCosStart)
      .addScaledVector(_slAxis2, d * link.__selfLoopSinStart)
      .add(_slStart);

    _slCp2.set(0, 0, 0)
      .addScaledVector(_slAxis1, d * link.__selfLoopCosEnd)
      .addScaledVector(_slAxis2, d * link.__selfLoopSinEnd)
      .add(_slStart);

    // Update the curve (reuse existing CubicBezierCurve3 — just move control points)
    const curve = link.__curve;
    if (curve) {
      curve.v0.copy(_slStart);
      curve.v1.copy(_slCp1);
      curve.v2.copy(_slCp2);
      curve.v3.copy(_slStart);
    }

    link.__cachedLength = null; // invalidate arrow length cache

    const line = lineObj.children.length ? lineObj.children[0] : lineObj;

    if (line.type === 'Line' && curve) {
      const curvePnts = curve.getPoints(SELF_LOOP_CURVE_RESOLUTION);
      if (line.geometry.getAttribute('position').array.length !== curvePnts.length * 3) {
        line.geometry[setAttributeFn]('position', new three.BufferAttribute(new Float32Array(curvePnts.length * 3), 3));
      }
      line.geometry.setFromPoints(curvePnts);
      line.geometry.computeBoundingSphere();
    } else if (line.type === 'Mesh' && curve) {
      const linkWidth = Math.ceil(linkWidthAccessor(link) * 10) / 10;
      const r = linkWidth / 2;
      const geometry = new three.TubeGeometry(curve, SELF_LOOP_CURVE_RESOLUTION, r, 3, false);
      line.geometry.dispose();
      line.geometry = geometry;
    }
  }

  // Only cache direction if all self-loops were updated. If some were skipped
  // (hidden), invalidate so the function re-runs when they become visible again.
  if (anySkipped) {
    _slLastCamDir.set(NaN, NaN, NaN);
  } else {
    _slLastCamDir.copy(_slCamDir);
  }
}

export default Kapsule({

  props: {
    jsonUrl: {
      onChange: function(jsonUrl, state) {
        if (jsonUrl && !state.fetchingJson) {
          // Load data asynchronously
          state.fetchingJson = true;
          state.onLoading();

          fetch(jsonUrl).then(r => r.json()).then(json => {
            state.fetchingJson = false;
            state.onFinishLoading(json);
            this.graphData(json);
          });
        }
      },
      triggerUpdate: false
    },
    graphData: {
      default: {
        nodes: [],
        links: []
      },
      onChange(graphData, state) {
        state.engineRunning = false; // Pause simulation immediately
        state._selfLoopLinks = []; // Clear stale self-loop refs from previous data
        _slLastCamDir.set(NaN, NaN, NaN); // Invalidate cache so billboard runs on first frame
      }
    },
    numDimensions: {
      default: 3,
      onChange(numDim, state) {
        const chargeForce = state.d3ForceLayout.force('charge');
        // Increase repulsion on 3D mode for improved spatial separation
        if (chargeForce) { chargeForce.strength(numDim > 2 ? -60 : -30) }

        if (numDim < 3) { eraseDimension(state.graphData.nodes, 'z'); }
        if (numDim < 2) { eraseDimension(state.graphData.nodes, 'y'); }

        function eraseDimension(nodes, dim) {
          nodes.forEach(d => {
            delete d[dim];     // position
            delete d[`v${dim}`]; // velocity
          });
        }
      }
    },
    dagMode: { onChange(dagMode, state) { // td, bu, lr, rl, zin, zout, radialin, radialout
      !dagMode && state.forceEngine === 'd3' && (state.graphData.nodes || []).forEach(n => n.fx = n.fy = n.fz = undefined); // unfix nodes when disabling dag mode
    }},
    dagLevelDistance: {},
    dagNodeFilter: { default: node => true },
    onDagError: { triggerUpdate: false },
    nodeRelSize: { default: 4 }, // volume per val unit
    nodeId: { default: 'id' },
    nodeVal: { default: 'val' },
    nodeResolution: { default: 8 }, // how many slice segments in the sphere's circumference
    nodeColor: { default: 'color' },
    nodeAutoColorBy: {},
    nodeOpacity: { default: 0.75 },
    nodeVisibility: { default: true },
    nodeThreeObject: {},
    nodeThreeObjectExtend: { default: false },
    nodePositionUpdate: { triggerUpdate: false }, // custom function to call for updating the node's position. Signature: (threeObj, { x, y, z}, node). If the function returns a truthy value, the regular node position update will not run.
    linkSource: { default: 'source' },
    linkTarget: { default: 'target' },
    linkVisibility: { default: true },
    linkColor: { default: 'color' },
    linkAutoColorBy: {},
    linkOpacity: { default: 0.2 },
    linkWidth: {}, // Rounded to nearest decimal. For falsy values use dimensionless line with 1px regardless of distance.
    linkResolution: { default: 6 }, // how many radial segments in each line tube's geometry
    linkCurvature: { default: 0, triggerUpdate: false }, // line curvature radius (0: straight, 1: semi-circle)
    linkCurveRotation: { default: 0, triggerUpdate: false }, // line curve rotation along the line axis (0: interection with XY plane, PI: upside down)
    linkMaterial: {},
    linkThreeObject: {},
    linkThreeObjectExtend: { default: false },
    linkPositionUpdate: { triggerUpdate: false }, // custom function to call for updating the link's position. Signature: (threeObj, { start: { x, y, z},  end: { x, y, z }}, link). If the function returns a truthy value, the regular link position update will not run.
    linkDirectionalArrowLength: { default: 0 },
    linkDirectionalArrowColor: {},
    linkDirectionalArrowRelPos: { default: 0.5, triggerUpdate: false }, // value between 0<>1 indicating the relative pos along the (exposed) line
    linkDirectionalArrowResolution: { default: 8 }, // how many slice segments in the arrow's conic circumference
    linkDirectionalParticles: { default: 0 }, // animate photons travelling in the link direction
    linkDirectionalParticleSpeed: { default: 0.01, triggerUpdate: false }, // in link length ratio per frame
    linkDirectionalParticleOffset: { default: 0, triggerUpdate: false }, // starting position offset along the link's length, like a pre-delay. Values between [0, 1]
    linkDirectionalParticleWidth: { default: 0.5 },
    linkDirectionalParticleColor: {},
    linkDirectionalParticleResolution: { default: 4 }, // how many slice segments in the particle sphere's circumference
    linkDirectionalParticleThreeObject: {},
    useInstancedRendering: { default: true }, // When false, all nodes/links go through ThreeDigest (individual meshes)
    forceEngine: { default: 'd3' }, // d3 or ngraph
    d3AlphaMin: { default: 0 },
    d3AlphaDecay: { default: 0.0228, triggerUpdate: false, onChange(alphaDecay, state) { state.d3ForceLayout.alphaDecay(alphaDecay) }},
    d3AlphaTarget: { default: 0, triggerUpdate: false, onChange(alphaTarget, state) { state.d3ForceLayout.alphaTarget(alphaTarget) }},
    d3VelocityDecay: { default: 0.4, triggerUpdate: false, onChange(velocityDecay, state) { state.d3ForceLayout.velocityDecay(velocityDecay) } },
    ngraphPhysics: { default: {
      // defaults from https://github.com/anvaka/ngraph.physics.simulator/blob/master/index.js
      timeStep: 20,
      gravity: -1.2,
      theta: 0.8,
      springLength: 30,
      springCoefficient: 0.0008,
      dragCoefficient: 0.02
    }},
    warmupTicks: { default: 0, triggerUpdate: false }, // how many times to tick the force engine at init before starting to render
    cooldownTicks: { default: Infinity, triggerUpdate: false },
    cooldownTime: { default: 15000, triggerUpdate: false }, // ms
    ticksPerFrame: { default: 10, triggerUpdate: false }, // how many simulation ticks to run per render frame (batch ticking)
    onLoading: { default: () => {}, triggerUpdate: false },
    onFinishLoading: { default: () => {}, triggerUpdate: false },
    onUpdate: { default: () => {}, triggerUpdate: false },
    onFinishUpdate: { default: () => {}, triggerUpdate: false },
    onEngineTick: { default: () => {}, triggerUpdate: false },
    onEngineStop: { default: () => {}, triggerUpdate: false }
  },

  methods: {
    refresh: function(state) {
      state._flushObjects = true;
      state._rerender();
      return this;
    },
    // Expose d3 forces for external manipulation
    d3Force: function(state, forceName, forceFn) {
      if (forceFn === undefined) {
        return state.d3ForceLayout.force(forceName); // Force getter
      }
      state.d3ForceLayout.force(forceName, forceFn); // Force setter
      return this;
    },
    d3ReheatSimulation: function(state) {
      state.d3ForceLayout.alpha(1);
      this.resetCountdown();
      return this;
    },
    // reset cooldown state
    resetCountdown: function(state) {
      state.cntTicks = 0;
      state.startTickTime = new Date();
      state.engineRunning = true;
      return this;
    },
    tickFrame: function(state, camera) {
      const isD3Sim = state.forceEngine !== 'ngraph';

      if (state.engineRunning) { layoutTick(); }

      // Instanced nodes: cheap updatePositions during simulation, sorted only when idle
      if (state.useInstancedRendering && state.instancedNodeRenderer.count > 0) {
        if (state.engineRunning) {
          // Simulation running — positions changing every frame, sort is wasted CPU
          state.instancedNodeRenderer.updatePositions(isD3Sim);
        } else {
          // Simulation stopped — sort by camera distance for correct transparency
          state.instancedNodeRenderer.sortAndUpdatePositions(camera, isD3Sim);
        }
      }

      // Billboard self-loops: recalculate curves to face camera when simulation is idle.
      // During simulation, layoutTick already handles curve + geometry updates.
      // This ensures loops stay camera-facing after simulation stops and user rotates.
      if (!state.engineRunning && camera) {
        updateSelfLoopBillboards(state, camera, isD3Sim);
      }

      updateArrows();
      updatePhotons();

      return this;

      //

      function layoutTick() {
        // Batch ticking: run multiple simulation ticks per render frame
        const batchSize = Math.max(1, state.ticksPerFrame || 1);
        let stopped = false;

        for (let t = 0; t < batchSize; t++) {
          if (
            ++state.cntTicks > state.cooldownTicks ||
            (new Date()) - state.startTickTime > state.cooldownTime ||
            (isD3Sim && state.d3AlphaMin > 0 && state.d3ForceLayout.alpha() < state.d3AlphaMin)
          ) {
            stopped = true;
            break;
          }
          state.layout[isD3Sim ? 'tick' : 'step'](); // Tick it
        }

        if (stopped) {
          state.engineRunning = false; // Stop ticking graph
          // Trigger one sort pass now that positions are final
          if (state.useInstancedRendering) {
            state.instancedNodeRenderer._positionsDirty = true;
          }
          state.onEngineStop();
        } else {
          state.onEngineTick();
        }

        // Update custom node positions (ThreeDigest path — cluster nodes etc.)
        const nodeThreeObjectExtendAccessor = accessorFn(state.nodeThreeObjectExtend);
        state.nodeDataMapper.entries().forEach(([node, obj]) => {
          if (!obj) return;

          const pos = isD3Sim ? node : state.layout.getNodePosition(node[state.nodeId]);

          const extendedObj = nodeThreeObjectExtendAccessor(node);
          if (!state.nodePositionUpdate
            || !state.nodePositionUpdate(extendedObj ? obj.children[0] : obj, { x: pos.x, y: pos.y, z: pos.z }, node)
            || extendedObj) {
            obj.position.x = pos.x;
            obj.position.y = pos.y || 0;
            obj.position.z = pos.z || 0;
          }
        });

        // Update instanced link positions
        const linkWidthAccessor = accessorFn(state.linkWidth);
        if (state.useInstancedRendering) {
          state.instancedLinkRenderer.updatePositions(isD3Sim, linkWidthAccessor(state.graphData.links[0] || {}));
        }

        // Update custom/curved link positions (ThreeDigest path)
        const linkCurvatureAccessor = accessorFn(state.linkCurvature);
        const linkCurveRotationAccessor = accessorFn(state.linkCurveRotation);
        const linkThreeObjectExtendAccessor = accessorFn(state.linkThreeObjectExtend);
        // Rebuild self-loop index for per-frame billboard update
        state._selfLoopLinks = [];
        state.linkDataMapper.entries().forEach(([link, lineObj]) => {
          if (!lineObj) return;

          // Skip hidden links entirely (e.g. self-edges hidden during camera movement)
          if (!lineObj.visible) return;

          const pos = isD3Sim
            ? link
            : state.layout.getLinkPosition(state.layout.graph.getLink(link.source, link.target).id);
          const start = pos[isD3Sim ? 'source' : 'from'];
          const end = pos[isD3Sim ? 'target' : 'to'];

          if (!start || !end || !start.hasOwnProperty('x') || !end.hasOwnProperty('x')) return; // skip invalid link

          calcLinkCurve(link); // calculate link curve for all links, including custom replaced, so it can be used in directional functionality
          link.__cachedLength = null; // invalidate arrow length cache

          const extendedObj = linkThreeObjectExtendAccessor(link);
          if (state.linkPositionUpdate && state.linkPositionUpdate(
              extendedObj ? lineObj.children[1] : lineObj, // pass child custom object if extending the default
              { start: { x: start.x, y: start.y, z: start.z }, end: { x: end.x, y: end.y, z: end.z } },
              link)
          && !extendedObj) {
            // exit if successfully custom updated position of non-extended obj
            return;
          }

          const isSelfLoop = link.__isSelfLoop;
          const curveResolution = isSelfLoop ? SELF_LOOP_CURVE_RESOLUTION : 30;
          const curve = link.__curve;

          // select default line obj if it's an extended group
          const line = lineObj.children.length ? lineObj.children[0] : lineObj;

          if (line.type === 'Line') { // Update line geometry
            if (!curve) { // straight line
              let linePos = line.geometry.getAttribute('position');
              if (!linePos || !linePos.array || linePos.array.length !== 6) {
                line.geometry[setAttributeFn]('position', linePos = new three.BufferAttribute(new Float32Array(2 * 3), 3));
              }

              linePos.array[0] = start.x;
              linePos.array[1] = start.y || 0;
              linePos.array[2] = start.z || 0;
              linePos.array[3] = end.x;
              linePos.array[4] = end.y || 0;
              linePos.array[5] = end.z || 0;

              linePos.needsUpdate = true;

            } else { // bezier curve line
              const curvePnts = curve.getPoints(curveResolution);
              // resize buffer if needed
              if (line.geometry.getAttribute('position').array.length !== curvePnts.length * 3) {
                line.geometry[setAttributeFn]('position', new three.BufferAttribute(new Float32Array(curvePnts.length * 3), 3));
              }
              line.geometry.setFromPoints(curvePnts);
            }
            line.geometry.computeBoundingSphere();

          } else if (line.type === 'Mesh') { // Update cylinder geometry

            if (!curve) { // straight tube
              if (!line.geometry.type.match(/^Cylinder(Buffer)?Geometry$/)) {
                const linkWidth = Math.ceil(linkWidthAccessor(link) * 10) / 10;
                const r = linkWidth / 2;

                const geometry = new three.CylinderGeometry(r, r, 1, state.linkResolution, 1, false);
                geometry[applyMatrix4Fn](new three.Matrix4().makeTranslation(0, 1 / 2, 0));
                geometry[applyMatrix4Fn](new three.Matrix4().makeRotationX(Math.PI / 2));

                line.geometry.dispose();
                line.geometry = geometry;
              }

              const vStart = new three.Vector3(start.x, start.y || 0, start.z || 0);
              const vEnd = new three.Vector3(end.x, end.y || 0, end.z || 0);
              const distance = vStart.distanceTo(vEnd);

              line.position.x = vStart.x;
              line.position.y = vStart.y;
              line.position.z = vStart.z;

              line.scale.z = distance;

              line.parent.localToWorld(vEnd); // lookAt requires world coords
              line.lookAt(vEnd);
            } else { // curved tube
              if (!line.geometry.type.match(/^Tube(Buffer)?Geometry$/)) {
                // reset object positioning
                line.position.set(0, 0, 0);
                line.rotation.set(0, 0, 0);
                line.scale.set(1, 1, 1);
              }

              const linkWidth = Math.ceil(linkWidthAccessor(link) * 10) / 10;
              const r = linkWidth / 2;
              const tubeRadialSegments = isSelfLoop ? 3 : state.linkResolution;

              const geometry = new three.TubeGeometry(curve, curveResolution, r, tubeRadialSegments, false);

              line.geometry.dispose();
              line.geometry = geometry;
            }
          }
        });

        //

        function calcLinkCurve(link) {
          const pos = isD3Sim
            ? link
            : state.layout.getLinkPosition(state.layout.graph.getLink(link.source, link.target).id);
          const start = pos[isD3Sim ? 'source' : 'from'];
          const end = pos[isD3Sim ? 'target' : 'to'];

          if (!start || !end || !start.hasOwnProperty('x') || !end.hasOwnProperty('x')) return; // skip invalid link

          const curvature = linkCurvatureAccessor(link);

          if (!curvature) {
            link.__curve = null; // Straight line
            link.__isSelfLoop = false;

          } else { // bezier curve line (only for line types)
            const vStart = new three.Vector3(start.x, start.y || 0, start.z || 0);
            const vEnd= new three.Vector3(end.x, end.y || 0, end.z || 0);

            const l = vStart.distanceTo(vEnd); // line length

            let curve;
            const curveRotation = linkCurveRotationAccessor(link);

            if (l > 0) {
              link.__isSelfLoop = false;
              const dx = end.x - start.x;
              const dy = end.y - start.y || 0;

              const vLine = new three.Vector3()
                .subVectors(vEnd, vStart);

              const cp = vLine.clone()
                .multiplyScalar(curvature)
                .cross((dx !== 0 || dy !== 0) ? new three.Vector3(0, 0, 1) : new three.Vector3(0, 1, 0)) // avoid cross-product of parallel vectors (prefer Z, fallback to Y)
                .applyAxisAngle(vLine.normalize(), curveRotation) // rotate along line axis according to linkCurveRotation
                .add((new three.Vector3()).addVectors(vStart, vEnd).divideScalar(2));

              curve = new three.QuadraticBezierCurve3(vStart, cp, vEnd);
            } else { // Same point — self-loop. Mark for billboard update every frame.
              // Compute node radius so the loop scales with node size
              const nodeValAccessor = accessorFn(state.nodeVal);
              const nodeR = Math.cbrt(Math.max(0, nodeValAccessor(start) || 1)) * state.nodeRelSize;
              const d = curvature * 70 + nodeR;

              // Store params for per-frame billboard recalc in updateSelfLoopBillboards()
              link.__isSelfLoop = true;
              state._selfLoopLinks.push(link);
              link.__selfLoopD = d;
              const sa = -curveRotation + Math.PI / 2;
              const ea = -curveRotation;
              // Pre-cache trig (constant per link — avoids recomputing every frame)
              link.__selfLoopCosStart = Math.cos(sa);
              link.__selfLoopSinStart = Math.sin(sa);
              link.__selfLoopCosEnd = Math.cos(ea);
              link.__selfLoopSinEnd = Math.sin(ea);

              // Initial curve (XY plane fallback — will be overwritten by billboard update)
              const cp1 = new three.Vector3(d * link.__selfLoopCosStart, d * link.__selfLoopSinStart, 0).add(vStart);
              const cp2 = new three.Vector3(d * link.__selfLoopCosEnd, d * link.__selfLoopSinEnd, 0).add(vStart);
              curve = new three.CubicBezierCurve3(vStart, cp1, cp2, vEnd);
            }

            link.__curve = curve;
          }
        }
      }

      function updateArrows() {
        // update instanced arrows
        const arrowRelPosAccessor = accessorFn(state.linkDirectionalArrowRelPos);
        const arrowLengthAccessor = accessorFn(state.linkDirectionalArrowLength);
        const nodeValAccessor = accessorFn(state.nodeVal);

        if (state.instancedArrowRenderer._created && state.instancedArrowRenderer.count > 0) {
          state.instancedArrowRenderer.updatePositions(isD3Sim, {
            arrowLength: arrowLengthAccessor,
            arrowRelPos: arrowRelPosAccessor,
            nodeVal: nodeValAccessor,
            nodeRelSize: state.nodeRelSize,
          });
        }

        // update custom arrows (ThreeDigest path)
        state.arrowDataMapper.entries().forEach(([link, arrowObj]) => {
          if (!arrowObj) return;

          // Skip arrows for hidden links (e.g. self-edges hidden during camera movement)
          const lineObj = link.__lineObj;
          if (lineObj && !lineObj.visible) {
            arrowObj.visible = false;
            return;
          }
          arrowObj.visible = true;

          const pos = isD3Sim
            ? link
            : state.layout.getLinkPosition(state.layout.graph.getLink(link.source, link.target).id);
          const start = pos[isD3Sim ? 'source' : 'from'];
          const end = pos[isD3Sim ? 'target' : 'to'];

          if (!start || !end || !start.hasOwnProperty('x') || !end.hasOwnProperty('x')) return; // skip invalid link

          const startR = Math.cbrt(Math.max(0, nodeValAccessor(start) || 1)) * state.nodeRelSize;
          const endR = Math.cbrt(Math.max(0, nodeValAccessor(end) || 1)) * state.nodeRelSize;

          const arrowLength = arrowLengthAccessor(link);
          const arrowRelPos = arrowRelPosAccessor(link);

          const getPosAlongLine = link.__curve
            ? t => link.__curve.getPoint(t) // interpolate along bezier curve
            : t => {
            // straight line: interpolate linearly
            const iplt = (dim, start, end, t) => start[dim] + (end[dim] - start[dim]) * t || 0;
            return {
              x: iplt('x', start, end, t),
              y: iplt('y', start, end, t),
              z: iplt('z', start, end, t)
            }
          };

          // Use cached length when available (invalidated in layoutTick/billboard update)
          const lineLen = link.__curve
            ? (link.__cachedLength ?? (link.__cachedLength = link.__curve.getLength()))
            : Math.sqrt(['x', 'y', 'z'].map(dim => Math.pow((end[dim] || 0) - (start[dim] || 0), 2)).reduce((acc, v) => acc + v, 0));

          const posAlongLine = startR + arrowLength + (lineLen - startR - endR - arrowLength) * arrowRelPos;

          const arrowHead = getPosAlongLine(posAlongLine / lineLen);
          const arrowTail = getPosAlongLine((posAlongLine - arrowLength) / lineLen);

          ['x', 'y', 'z'].forEach(dim => arrowObj.position[dim] = arrowTail[dim]);

          const headVec = new three.Vector3(...['x', 'y', 'z'].map(c => arrowHead[c]));
          arrowObj.parent.localToWorld(headVec); // lookAt requires world coords
          arrowObj.lookAt(headVec);
        });
      }

      function updatePhotons() {
        // update link particle positions
        const particleSpeedAccessor = accessorFn(state.linkDirectionalParticleSpeed);
        const particleOffsetAccessor = accessorFn(state.linkDirectionalParticleOffset);
        state.graphData.links.forEach(link => {
          const photonsObj = state.particlesDataMapper.getObj(link);
          const cyclePhotons = photonsObj && photonsObj.children;
          const singleHopPhotons = link.__singleHopPhotonsObj && link.__singleHopPhotonsObj.children;

          if ((!singleHopPhotons || !singleHopPhotons.length) && (!cyclePhotons || !cyclePhotons.length)) return;

          const pos = isD3Sim
            ? link
            : state.layout.getLinkPosition(state.layout.graph.getLink(link.source, link.target).id);
          const start = pos[isD3Sim ? 'source' : 'from'];
          const end = pos[isD3Sim ? 'target' : 'to'];

          if (!start || !end || !start.hasOwnProperty('x') || !end.hasOwnProperty('x')) return; // skip invalid link

          const particleSpeed = particleSpeedAccessor(link);
          const particleOffset = Math.abs(particleOffsetAccessor(link));

          const getPhotonPos = link.__curve
            ? t => link.__curve.getPoint(t) // interpolate along bezier curve
            : t => {
              // straight line: interpolate linearly
              const iplt = (dim, start, end, t) => start[dim] + (end[dim] - start[dim]) * t || 0;
              return {
                x: iplt('x', start, end, t),
                y: iplt('y', start, end, t),
                z: iplt('z', start, end, t)
              }
            };

          const photons = [...(cyclePhotons || []), ...(singleHopPhotons || []),];

          photons.forEach((photon, idx) => {
            const singleHop = photon.parent.__linkThreeObjType === 'singleHopPhotons';

            if (!photon.hasOwnProperty('__progressRatio')) {
              photon.__progressRatio = singleHop ? 0 : ((idx + particleOffset) / cyclePhotons.length);
            }

            photon.__progressRatio += particleSpeed;

            if (photon.__progressRatio >=1) {
              if (!singleHop) {
                photon.__progressRatio = photon.__progressRatio % 1;
              } else {
                // remove particle
                photon.parent.remove(photon);
                emptyObject(photon);
                return;
              }
            }

            const photonPosRatio = photon.__progressRatio;

            const pos = getPhotonPos(photonPosRatio);

            // Orient asymmetrical particles to target
            photon.geometry.type !== 'SphereGeometry' && photon.lookAt(pos.x, pos.y, pos.z);

            ['x', 'y', 'z'].forEach(dim => photon.position[dim] = pos[dim]);
          });
        });
      }
    },
    emitParticle: function(state, link) {
      if (link && state.graphData.links.includes(link)) {
        if (!link.__singleHopPhotonsObj) {
          const obj = new three.Group();
          obj.__linkThreeObjType = 'singleHopPhotons';
          link.__singleHopPhotonsObj = obj;

          state.graphScene.add(obj);
        }

        let particleObj = accessorFn(state.linkDirectionalParticleThreeObject)(link);
        if (particleObj && state.linkDirectionalParticleThreeObject === particleObj) {
          // clone object if it's a shared object among all links
          particleObj = particleObj.clone();
        }

        if (!particleObj) {
          const particleWidthAccessor = accessorFn(state.linkDirectionalParticleWidth);
          const photonR = Math.ceil(particleWidthAccessor(link) * 10) / 10 / 2;
          const numSegments = state.linkDirectionalParticleResolution;
          const particleGeometry = new three.SphereGeometry(photonR, numSegments, numSegments);

          const linkColorAccessor = accessorFn(state.linkColor);
          const particleColorAccessor = accessorFn(state.linkDirectionalParticleColor);
          const photonColor = particleColorAccessor(link) || linkColorAccessor(link) || '#f0f0f0';
          const materialColor = new three.Color(colorStr2Hex(photonColor));
          const opacity = state.linkOpacity * 3;
          const particleMaterial = new three.MeshLambertMaterial({
            color: materialColor,
            transparent: true,
            opacity
          });

          particleObj = new three.Mesh(particleGeometry, particleMaterial);
        }

        // add a single hop particle
        link.__singleHopPhotonsObj.add(particleObj);
      }

      return this;
    },
    getGraphBbox: function(state, nodeFilter = () => true) {
      if (!state.initialised) return null;

      // Collect bboxes from instanced renderers
      const bboxes = [];
      const instancedNodeBbox = state.instancedNodeRenderer.computeBBox(nodeFilter);
      if (instancedNodeBbox) bboxes.push(instancedNodeBbox);
      const instancedLinkBbox = state.instancedLinkRenderer.computeBBox();
      if (instancedLinkBbox) bboxes.push(instancedLinkBbox);

      // Also collect from ThreeDigest objects (custom nodes/links)
      ;(function getBboxes(obj) {
        // Skip instanced renderer meshes (already handled above)
        if (obj.__isInstancedRenderer) return;

        if (obj.geometry && !obj.isInstancedMesh) {
          obj.geometry.computeBoundingBox();
          const box = new three.Box3();
          box.copy(obj.geometry.boundingBox).applyMatrix4(obj.matrixWorld);
          bboxes.push(box);
        }
        (obj.children || [])
          .filter(obj => !obj.hasOwnProperty('__graphObjType') ||
            (obj.__graphObjType === 'node' && nodeFilter(obj.__data))
          )
          .forEach(getBboxes);
      })(state.graphScene);

      if (!bboxes.length) return null;

      // extract global x,y,z min/max
      return Object.assign(...['x', 'y', 'z'].map(c => ({
        [c]: [
          d3Min(bboxes, bb => bb.min[c]),
          d3Max(bboxes, bb => bb.max[c])
        ]
      })));
    }
  },

  stateInit: () => ({
    d3ForceLayout: d3ForceSimulation()
      .force('link', d3ForceLink())
      .force('charge', d3ForceManyBody())
      .force('center', d3ForceCenter())
      .force('dagRadial', null)
      .stop(),
    engineRunning: false
  }),

  init(threeObj, state) {
    // Main three object to manipulate
    state.graphScene = threeObj;

    // Instanced renderers for default nodes/links/arrows (single draw call each)
    state.instancedNodeRenderer = new InstancedNodeRenderer(threeObj);
    state.instancedNodeRenderer.init(state.nodeResolution);
    state.instancedLinkRenderer = new InstancedLinkRenderer(threeObj);
    state.instancedArrowRenderer = new InstancedArrowRenderer(threeObj);

    // ThreeDigest kept only for custom objects (nodeThreeObject, curved links, etc.)
    state.nodeDataMapper = new ThreeDigest(threeObj, { objBindAttr: '__threeObj' });
    state.linkDataMapper = new ThreeDigest(threeObj, { objBindAttr: '__lineObj' });
    state.arrowDataMapper = new ThreeDigest(threeObj, { objBindAttr: '__arrowObj' });
    state.particlesDataMapper = new ThreeDigest(threeObj, { objBindAttr: '__photonsObj' });
  },

  update(state, changedProps) {
    const _t0 = performance.now();
    const hasAnyPropChanged = propList => propList.some(p => changedProps.hasOwnProperty(p));

    state.engineRunning = false; // pause simulation
    (typeof state.onUpdate === "function") && state.onUpdate();

    if (state.nodeAutoColorBy !== null && hasAnyPropChanged(['nodeAutoColorBy', 'graphData', 'nodeColor'])) {
      // Auto add color to uncolored nodes
      autoColorObjects(state.graphData.nodes, accessorFn(state.nodeAutoColorBy), state.nodeColor);
    }
    if (state.linkAutoColorBy !== null && hasAnyPropChanged(['linkAutoColorBy', 'graphData', 'linkColor'])) {
      // Auto add color to uncolored links
      autoColorObjects(state.graphData.links, accessorFn(state.linkAutoColorBy), state.linkColor);
    }

    // Digest nodes WebGL objects
    if (state._flushObjects || hasAnyPropChanged([
      'graphData',
      'nodeThreeObject',
      'nodeThreeObjectExtend',
      'nodeVal',
      'nodeColor',
      'nodeVisibility',
      'nodeRelSize',
      'nodeResolution',
      'nodeOpacity',
      'useInstancedRendering'
    ])) {
      const customObjectAccessor = accessorFn(state.nodeThreeObject);
      const customObjectExtendAccessor = accessorFn(state.nodeThreeObjectExtend);
      const valAccessor = accessorFn(state.nodeVal);
      const colorAccessor = accessorFn(state.nodeColor);
      const visibilityAccessor = accessorFn(state.nodeVisibility);

      const visibleNodes = state.graphData.nodes.filter(visibilityAccessor);

      // Partition nodes: custom (nodeThreeObject returns truthy) vs default (instanced)
      const customNodes = [];
      const defaultNodes = [];
      if (state.useInstancedRendering) {
        visibleNodes.forEach(node => {
          if (customObjectAccessor(node)) {
            customNodes.push(node);
          } else {
            defaultNodes.push(node);
          }
        });
      } else {
        // All nodes go through ThreeDigest when instanced rendering is disabled
        customNodes.push(...visibleNodes);
      }

      // === Instanced path for default nodes (single draw call) ===
      if (state.useInstancedRendering) {
        // Re-init if resolution changed
        if (state.instancedNodeRenderer._resolution !== state.nodeResolution) {
          state.instancedNodeRenderer.init(state.nodeResolution);
        }
        state.instancedNodeRenderer.digest(defaultNodes, {
          val: valAccessor,
          color: colorAccessor,
          opacity: state.nodeOpacity,
          relSize: state.nodeRelSize,
        });
      } else {
        // Clear instanced renderer
        if (state.instancedNodeRenderer._created) {
          state.instancedNodeRenderer.mesh.count = 0;
          state.instancedNodeRenderer.count = 0;
        }
      }

      // === ThreeDigest path for custom nodes only (cluster shapes, etc.) ===
      const sphereGeometries = {}; // indexed by node value
      const sphereMaterials = {}; // indexed by color

      if (state._flushObjects || hasAnyPropChanged([
        'nodeThreeObject',
        'nodeThreeObjectExtend'
      ])) state.nodeDataMapper.clear();

      state.nodeDataMapper
        .onCreateObj(node => {
          let customObj = customObjectAccessor(node);
          const extendObj = customObjectExtendAccessor(node);

          if (customObj && state.nodeThreeObject === customObj) {
            customObj = customObj.clone();
          }

          let obj;

          if (customObj && !extendObj) {
            obj = customObj;
          } else {
            obj = new three.Mesh();
            obj.__graphDefaultObj = true;

            if (customObj && extendObj) {
              obj.add(customObj);
            }
          }

          obj.__graphObjType = 'node';

          return obj;
        })
        .onUpdateObj((obj, node) => {
          if (obj.__graphDefaultObj) {
            const val = valAccessor(node) || 1;
            const radius = Math.cbrt(val) * state.nodeRelSize;
            const numSegments = state.nodeResolution;

            if (!obj.geometry.type.match(/^Sphere(Buffer)?Geometry$/)
              || obj.geometry.parameters.radius !== radius
              || obj.geometry.parameters.widthSegments !== numSegments
            ) {
              if (!sphereGeometries.hasOwnProperty(val)) {
                sphereGeometries[val] = new three.SphereGeometry(radius, numSegments, numSegments);
              }

              obj.geometry.dispose();
              obj.geometry = sphereGeometries[val];
            }

            const color = colorAccessor(node);
            const materialColor = new three.Color(colorStr2Hex(color || '#ffffaa'));
            const opacity = state.nodeOpacity * colorAlpha(color);

            if (obj.material.type !== 'MeshLambertMaterial'
              || !obj.material.color.equals(materialColor)
              || obj.material.opacity !== opacity
            ) {
              if (!sphereMaterials.hasOwnProperty(color)) {
                sphereMaterials[color] = new three.MeshLambertMaterial({
                  color: materialColor,
                  transparent: true,
                  opacity
                });
              }

              obj.material.dispose();
              obj.material = sphereMaterials[color];
            }
          }
        })
        .digest(customNodes);
    }

    // Digest links WebGL objects
    if (state._flushObjects || hasAnyPropChanged([
      'graphData',
      'linkThreeObject',
      'linkThreeObjectExtend',
      'linkMaterial',
      'linkColor',
      'linkWidth',
      'linkVisibility',
      'linkResolution',
      'linkOpacity',
      'linkDirectionalArrowLength',
      'linkDirectionalArrowColor',
      'linkDirectionalArrowResolution',
      'linkDirectionalParticles',
      'linkDirectionalParticleWidth',
      'linkDirectionalParticleColor',
      'linkDirectionalParticleResolution',
      'linkDirectionalParticleThreeObject',
      'useInstancedRendering'
    ])) {
      const customObjectAccessor = accessorFn(state.linkThreeObject);
      const customObjectExtendAccessor = accessorFn(state.linkThreeObjectExtend);
      const customMaterialAccessor = accessorFn(state.linkMaterial);
      const visibilityAccessor = accessorFn(state.linkVisibility);
      const colorAccessor = accessorFn(state.linkColor);
      const widthAccessor = accessorFn(state.linkWidth);
      const linkCurvatureAccessor = accessorFn(state.linkCurvature);

      const visibleLinks = state.graphData.links.filter(visibilityAccessor);

      // Partition links: custom/curved → ThreeDigest, default/straight → instanced
      const customLinks = [];
      const defaultLinks = [];
      if (state.useInstancedRendering) {
        visibleLinks.forEach(link => {
          if (customObjectAccessor(link) || linkCurvatureAccessor(link)) {
            customLinks.push(link);
          } else {
            defaultLinks.push(link);
          }
        });
      } else {
        // All links go through ThreeDigest when instanced rendering is disabled
        customLinks.push(...visibleLinks);
      }

      // === Instanced path for default straight links (single draw call) ===
      if (state.useInstancedRendering) {
        const useCylinder = !!widthAccessor(defaultLinks[0] || {});
        // Initialize or re-init if mode changed
        if (!state.instancedLinkRenderer._created
          || (useCylinder && state.instancedLinkRenderer._mode !== 'cylinder')
          || (!useCylinder && state.instancedLinkRenderer._mode !== 'line')
        ) {
          state.instancedLinkRenderer.init(useCylinder, state.linkResolution);
        }

        state.instancedLinkRenderer.digest(defaultLinks, {
          color: colorAccessor,
          opacity: state.linkOpacity,
          width: widthAccessor,
        });
      } else {
        // Clear instanced link renderer
        if (state.instancedLinkRenderer._created) {
          if (state.instancedLinkRenderer._mode === 'line' && state.instancedLinkRenderer.lineSegments) {
            state.instancedLinkRenderer.lineSegments.geometry.setDrawRange(0, 0);
          } else if (state.instancedLinkRenderer.cylinderMesh) {
            state.instancedLinkRenderer.cylinderMesh.count = 0;
          }
          state.instancedLinkRenderer.count = 0;
        }
      }

      // === Instanced arrows ===
      if (state.linkDirectionalArrowLength || changedProps.hasOwnProperty('linkDirectionalArrowLength')) {
        const arrowLengthAccessor = accessorFn(state.linkDirectionalArrowLength);
        const arrowColorAccessor = accessorFn(state.linkDirectionalArrowColor);

        if (state.useInstancedRendering) {
          if (!state.instancedArrowRenderer._created || state.instancedArrowRenderer._resolution !== state.linkDirectionalArrowResolution) {
            state.instancedArrowRenderer.init(state.linkDirectionalArrowResolution);
          }

          // Arrows on instanced links
          const instancedArrowLinks = defaultLinks.filter(arrowLengthAccessor);
          state.instancedArrowRenderer.digest(instancedArrowLinks, {
            arrowLength: arrowLengthAccessor,
            arrowColor: arrowColorAccessor,
            linkColor: colorAccessor,
            linkOpacity: state.linkOpacity,
          });
        } else {
          // Clear instanced arrow renderer
          if (state.instancedArrowRenderer._created) {
            state.instancedArrowRenderer.mesh.count = 0;
          }
        }

        // Arrows on custom links (ThreeDigest path) — all links when instanced is off
        const arrowSourceLinks = state.useInstancedRendering ? customLinks : visibleLinks;
        const customArrowLinks = arrowSourceLinks.filter(arrowLengthAccessor);
        state.arrowDataMapper
          .onCreateObj(() => {
            const obj = new three.Mesh(undefined, new three.MeshLambertMaterial({ transparent: true }));
            obj.__linkThreeObjType = 'arrow';
            return obj;
          })
          .onUpdateObj((obj, link) => {
            const arrowLength = arrowLengthAccessor(link);
            const numSegments = state.linkDirectionalArrowResolution;

            if (!obj.geometry.type.match(/^Cone(Buffer)?Geometry$/)
              || obj.geometry.parameters.height !== arrowLength
              || obj.geometry.parameters.radialSegments !== numSegments
            ) {
              const coneGeometry = new three.ConeGeometry(arrowLength * 0.25, arrowLength, numSegments);
              coneGeometry.translate(0, arrowLength / 2, 0);
              coneGeometry.rotateX(Math.PI / 2);
              obj.geometry.dispose();
              obj.geometry = coneGeometry;
            }

            const arrowColor = arrowColorAccessor(link) || colorAccessor(link) || '#f0f0f0';
            obj.material.color = new three.Color(colorStr2Hex(arrowColor));
            obj.material.opacity = state.linkOpacity * 3 * colorAlpha(arrowColor);
          })
          .digest(customArrowLinks);
      } else {
        // No arrows — clear instanced arrow renderer
        if (state.instancedArrowRenderer._created) {
          state.instancedArrowRenderer.mesh.count = 0;
        }
        state.arrowDataMapper.digest([]);
      }

      // === ThreeDigest path for custom/curved links ===
      const cylinderGeometries = {};
      const lambertLineMaterials = {};
      const basicLineMaterials = {};

      if (state._flushObjects || hasAnyPropChanged([
        'linkThreeObject',
        'linkThreeObjectExtend',
        'linkWidth'
      ])) state.linkDataMapper.clear();

      state.linkDataMapper
        .onRemoveObj(obj => {
          const singlePhotonsObj = obj.__data && obj.__data.__singleHopPhotonsObj;
          if (singlePhotonsObj) {
            singlePhotonsObj.parent.remove(singlePhotonsObj);
            emptyObject(singlePhotonsObj);
            delete obj.__data.__singleHopPhotonsObj;
          }
        })
        .onCreateObj(link => {
          let customObj = customObjectAccessor(link);
          const extendObj = customObjectExtendAccessor(link);

          if (customObj && state.linkThreeObject === customObj) {
            customObj = customObj.clone();
          }

          let defaultObj;
          if (!customObj || extendObj) {
            const useCylinder = !!widthAccessor(link);
            if (useCylinder) {
              defaultObj = new three.Mesh();
            } else {
              const lineGeometry = new three.BufferGeometry();
              lineGeometry[setAttributeFn]('position', new three.BufferAttribute(new Float32Array(2 * 3), 3));
              defaultObj = new three.Line(lineGeometry);
            }
          }

          let obj;
          if (!customObj) {
            obj = defaultObj;
            obj.__graphDefaultObj = true;
          } else {
            if (!extendObj) {
              obj = customObj;
            } else {
              obj = new three.Group();
              obj.__graphDefaultObj = true;
              obj.add(defaultObj);
              obj.add(customObj);
            }
          }

          obj.renderOrder = 10;
          obj.__graphObjType = 'link';
          return obj;
        })
        .onUpdateObj((updObj, link) => {
          if (updObj.__graphDefaultObj) {
            const obj = updObj.children.length ? updObj.children[0] : updObj;
            const linkWidth = Math.ceil(widthAccessor(link) * 10) / 10;
            const useCylinder = !!linkWidth;

            if (useCylinder) {
              const r = linkWidth / 2;
              const numSegments = state.linkResolution;

              if (!obj.geometry.type.match(/^Cylinder(Buffer)?Geometry$/)
                || obj.geometry.parameters.radiusTop !== r
                || obj.geometry.parameters.radialSegments !== numSegments
              ) {
                if (!cylinderGeometries.hasOwnProperty(linkWidth)) {
                  const geometry = new three.CylinderGeometry(r, r, 1, numSegments, 1, false);
                  geometry[applyMatrix4Fn](new three.Matrix4().makeTranslation(0, 1 / 2, 0));
                  geometry[applyMatrix4Fn](new three.Matrix4().makeRotationX(Math.PI / 2));
                  cylinderGeometries[linkWidth] = geometry;
                }
                obj.geometry.dispose();
                obj.geometry = cylinderGeometries[linkWidth];
              }
            }

            const customMaterial = customMaterialAccessor(link);
            if (customMaterial) {
              obj.material = customMaterial;
            } else {
              const color = colorAccessor(link);
              const materialColor = new three.Color(colorStr2Hex(color || '#f0f0f0'));
              const opacity = state.linkOpacity * colorAlpha(color);

              const materialType = useCylinder ? 'MeshLambertMaterial' : 'LineBasicMaterial';
              if (obj.material.type !== materialType
                || !obj.material.color.equals(materialColor)
                || obj.material.opacity !== opacity
              ) {
                const lineMaterials = useCylinder ? lambertLineMaterials : basicLineMaterials;
                if (!lineMaterials.hasOwnProperty(color)) {
                  lineMaterials[color] = new three[materialType]({
                    color: materialColor,
                    transparent: opacity < 1,
                    opacity,
                    depthWrite: opacity >= 1
                  });
                }
                obj.material.dispose();
                obj.material = lineMaterials[color];
              }
            }
          }
        })
        .digest(customLinks)

      // Photon particles digest cycle
      if (state.linkDirectionalParticles || changedProps.hasOwnProperty('linkDirectionalParticles')) {
        const particlesAccessor = accessorFn(state.linkDirectionalParticles);
        const particleWidthAccessor = accessorFn(state.linkDirectionalParticleWidth);
        const particleColorAccessor = accessorFn(state.linkDirectionalParticleColor);
        const particleObjectAccessor = accessorFn(state.linkDirectionalParticleThreeObject);

        const particleMaterials = {}; // indexed by link color
        const particleGeometries = {}; // indexed by particle width

        state.particlesDataMapper
          .onCreateObj(() => {
            const obj = new three.Group();
            obj.__linkThreeObjType = 'photons'; // Add object type

            obj.__photonDataMapper = new ThreeDigest(obj);

            return obj;
          })
          .onUpdateObj((obj, link) => {
            const curPhoton = !!obj.children.length && obj.children[0];
            const customObj = particleObjectAccessor(link);

            let particleGeometry, particleMaterial;
            if (customObj) {
              particleGeometry = customObj.geometry;
              particleMaterial = customObj.material;
            } else {
              const photonR = Math.ceil(particleWidthAccessor(link) * 10) / 10 / 2;
              const numSegments = state.linkDirectionalParticleResolution;

              if (curPhoton
                && curPhoton.geometry.parameters.radius === photonR
                && curPhoton.geometry.parameters.widthSegments === numSegments) {
                particleGeometry = curPhoton.geometry;
              } else {
                if (!particleGeometries.hasOwnProperty(photonR)) {
                  particleGeometries[photonR] = new three.SphereGeometry(photonR, numSegments, numSegments);
                }
                particleGeometry = particleGeometries[photonR];
              }

              const photonColor = particleColorAccessor(link) || colorAccessor(link) || '#f0f0f0';
              const materialColor = new three.Color(colorStr2Hex(photonColor));
              const opacity = state.linkOpacity * 3;

              if (curPhoton
                && curPhoton.material.color.equals(materialColor)
                && curPhoton.material.opacity === opacity
              ) {
                particleMaterial = curPhoton.material;
              } else {
                if (!particleMaterials.hasOwnProperty(photonColor)) {
                  particleMaterials[photonColor] = new three.MeshLambertMaterial({
                    color: materialColor,
                    transparent: true,
                    opacity
                  });
                }
                particleMaterial = particleMaterials[photonColor];
              }
            }

            if (curPhoton) {
              // Dispose of previous particles
              curPhoton.geometry !== particleGeometry && curPhoton.geometry.dispose();
              curPhoton.material !== particleMaterial && curPhoton.material.dispose();
            }

            // digest cycle for each photon
            const numPhotons = Math.round(Math.abs(particlesAccessor(link)));
            obj.__photonDataMapper
              .id(d => d.idx)
              .onCreateObj(() => new three.Mesh(particleGeometry, particleMaterial))
              .onUpdateObj(obj => {
                obj.geometry = particleGeometry;
                obj.material = particleMaterial;
              })
              .digest([...new Array(numPhotons)].map((_, idx) => ({ idx })));
          })
          .digest(visibleLinks.filter(particlesAccessor));
      }
    }

    state._flushObjects = false; // reset objects refresh flag

    // simulation engine
    if (hasAnyPropChanged([
      'graphData',
      'nodeId',
      'linkSource',
      'linkTarget',
      'numDimensions',
      'forceEngine',
      'dagMode',
      'dagNodeFilter',
      'dagLevelDistance'
    ])) {
      state.engineRunning = false; // Pause simulation

      // parse links
      state.graphData.links.forEach(link => {
        link.source = link[state.linkSource];
        link.target = link[state.linkTarget];
      });

      // Feed data to force-directed layout
      const isD3Sim = state.forceEngine !== 'ngraph';
      let layout;
      if (isD3Sim) {
        // D3-force
        (layout = state.d3ForceLayout)
          .stop()
          .alpha(1)// re-heat the simulation
          .numDimensions(state.numDimensions)
          .nodes(state.graphData.nodes);

        // add links (if link force is still active)
        const linkForce = state.d3ForceLayout.force('link');
        if (linkForce) {
          linkForce
            .id(d => d[state.nodeId])
            .links(state.graphData.links);
        }

        // setup dag force constraints
        const nodeDepths = state.dagMode && getDagDepths(
          state.graphData,
          node => node[state.nodeId],
          {
            nodeFilter: state.dagNodeFilter,
            onLoopError: state.onDagError || undefined
          }
        );
        const maxDepth = Math.max(...Object.values(nodeDepths || []));
        const dagLevelDistance = state.dagLevelDistance || (
          state.graphData.nodes.length / (maxDepth || 1) * DAG_LEVEL_NODE_RATIO
          * (['radialin', 'radialout'].indexOf(state.dagMode) !== -1 ? 0.7 : 1)
        );

        // Reset relevant f* when swapping dag modes
        if (['lr', 'rl', 'td', 'bu', 'zin', 'zout'].includes(changedProps.dagMode)) {
          const resetProp = ['lr', 'rl'].includes(changedProps.dagMode) ? 'fx' : ['td', 'bu'].includes(changedProps.dagMode) ? 'fy' : 'fz';
          state.graphData.nodes.filter(state.dagNodeFilter).forEach(node => delete node[resetProp]);
        }

        // Fix nodes to x,y,z for dag mode
        if (['lr', 'rl', 'td', 'bu', 'zin', 'zout'].includes(state.dagMode)) {
          const invert = ['rl', 'td', 'zout'].includes(state.dagMode);
          const fixFn = node => (nodeDepths[node[state.nodeId]] - maxDepth / 2) * dagLevelDistance * (invert ? -1 : 1);

          const resetProp = ['lr', 'rl'].includes(state.dagMode) ? 'fx' : ['td', 'bu'].includes(state.dagMode) ? 'fy' : 'fz';
          state.graphData.nodes.filter(state.dagNodeFilter).forEach(node => node[resetProp] = fixFn(node));
        }

        // Use radial force for radial dags
        state.d3ForceLayout.force('dagRadial',
          ['radialin', 'radialout'].indexOf(state.dagMode) !== -1
            ? d3ForceRadial(node => {
                const nodeDepth = nodeDepths[node[state.nodeId]] || -1;
                return (state.dagMode === 'radialin' ? maxDepth - nodeDepth : nodeDepth) * dagLevelDistance;
              })
              .strength(node => state.dagNodeFilter(node) ? 1 : 0)
            : null
        );
      } else {
        // ngraph
        const graph = ngraph.graph();
        state.graphData.nodes.forEach(node => { graph.addNode(node[state.nodeId]); });
        state.graphData.links.forEach(link => { graph.addLink(link.source, link.target); });
        layout = ngraph.forcelayout(graph, { dimensions: state.numDimensions, ...state.ngraphPhysics });
        layout.graph = graph; // Attach graph reference to layout
      }

      for (
        let i = 0;
        i < state.warmupTicks && !(isD3Sim && state.d3AlphaMin > 0 && state.d3ForceLayout.alpha() < state.d3AlphaMin);
        i++
      ) {
        layout[isD3Sim ? "tick" : "step"]();
      } // Initial ticks before starting to render

      state.layout = layout;
      this.resetCountdown();
    }

    state.engineRunning = true; // resume simulation

    if (typeof window !== 'undefined' && window.__PERF_METRICS__) {
      window.__PERF_METRICS__.entries.push({
        label: 'forcegraphUpdate',
        ms: performance.now() - _t0,
        ts: Date.now(),
        extra: { changedPropsCount: Object.keys(changedProps).length },
      });
    }
    state.onFinishUpdate();
  }
});
