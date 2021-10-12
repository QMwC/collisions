const WORLD_SIZE = [100, 100, 100];
const CELLS = WORLD_SIZE.reduce((a, b) => a*b);
const PARTICLES = 20000;

let spheresOrPoints = true;

let canvas = new Canvas(document.querySelector("canvas"));
let camera = new OrbitCamera(0, 0, 60, .1, 400, .95, _, [.001, 200]);

let sphere;
let pointVertices, pointPrgm;

if (spheresOrPoints) {
  sphere = new P2D_Extra(`
    float texIdx = floor(idx/(2.*18.*9.));
    u = u/8.*PI;
    v = mod(v, 9.)/8.*PI;
    vec2 texCoord = vec2(mod(texIdx, float0)+.5, floor(texIdx/float0)+.5)/vec2(float0, float1);
    return vector0.xyz+texture2D(dataTex, texCoord).rgb+.5*vec3(sin(v)*cos(u), sin(v)*sin(u), cos(v));
    `, `
    u = u/8.*PI;
    v = mod(v, 9.)/8.*PI;
    return vec3(sin(v)*cos(u), sin(v)*sin(u), cos(v));`,
     0, 16, 1, 0, PARTICLES*9, 1, [1, 1, 0]);
  sphere.setVector(0, WORLD_SIZE.over(2).times(-1).concat(0));
} else {
  pointPrgm = new Program(
    `precision mediump float;

    attribute float idx;

    uniform sampler2D dataTex;
    uniform vec2 texDim;

    uniform vec3 offset;

    uniform vec2 coordMult;
    uniform mat3 matrix;
    uniform vec3 camera;
    uniform float near, far;

    varying float vIdx;

    vec4 proj3Dto2D(vec3 v) {
      vec3 new = (v-camera)*matrix;
      return vec4(new.xy*coordMult, 2.*(1.-new.z/near)/(1./far-1./near)-new.z, new.z);
    }

    vec3 pos() {
      return offset+texture2D(dataTex, vec2(mod(idx, texDim.x)+.5, floor(idx/texDim.x)+.5)/texDim).rgb;
    }

    void main() {
      vIdx = idx;
      vec4 position = proj3Dto2D(pos());
      gl_Position = position;
      gl_PointSize = 200./position.z;
    }`,
    `precision mediump float;

    varying float vIdx;

    void main() {
      float i = mod(vIdx, 1000.);
      float r = i/1000.;
      i = mod(i, 100.);
      float g = i/100.;
      i = mod(i, 10.);
      float b = i/10.;
      gl_FragColor = vec4(r, g, b, 1);
    }`
  );

  GLOBAL_GL.uniform3fv(pointPrgm.uniformLocationOf("offset"), WORLD_SIZE.over(2).times(-1));

  PARAMETRIC_BUFFER = GLOBAL_GL.createBuffer();
  GLOBAL_GL.bindBuffer(GLOBAL_GL.ARRAY_BUFFER, PARAMETRIC_BUFFER);
  GLOBAL_GL.bufferData(GLOBAL_GL.ARRAY_BUFFER, new Float32Array(Array(PARTICLES).keys()), GLOBAL_GL.STATIC_DRAW);

  pointVertices = new Vertices(PARTICLES);
  pointVertices.defaultDrawMode = GLOBAL_GL.POINTS;
  pointVertices.addAttributes(
    pointPrgm,
    ["idx"],
    [1],
    [GLOBAL_GL.FLOAT],
    [0]
  );
}

let positionParams = {
  prgm: new Program(
    `precision mediump float;

    attribute vec2 pos;
    varying vec2 texPos;

    void main() {
      texPos = .5*(1.+pos);
      gl_Position = vec4(pos, 0, 1);
    }`,
    `precision mediump float;

    uniform sampler2D spacial;
    uniform sampler2D position;
    varying vec2 texPos;

    uniform vec3 worldSize;
    uniform vec2 spacialDim;
    uniform vec2 positionDim;

    vec2 posToSpacial(vec3 pos) {
      float idx = pos.x*worldSize.y*worldSize.z+pos.y*worldSize.z+pos.z;
      return vec2(mod(idx, spacialDim.x)+.5, floor(idx/spacialDim.x)+.5)/spacialDim;
    }

    vec3 spacialToPos(vec4 spcial) {
      float idx = spcial.a;
      return texture2D(position, vec2(mod(idx, positionDim.x)+.5, floor(idx/positionDim.x)+.5)/positionDim).rgb;
    }

    vec4 worldToSpacial(vec3 pos) {
      float idx = pos.x*worldSize.y*worldSize.z+pos.y*worldSize.z+pos.z;
      return texture2D(spacial, vec2(mod(idx, spacialDim.x)+.5, floor(idx/spacialDim.x)+.5)/spacialDim);
    }

    void main() {
      vec3 position = texture2D(position, texPos).rgb;
      vec3 floorPos = floor(position);
      vec2 spacialCoord = posToSpacial(floorPos);
      vec4 currSpacial = texture2D(spacial, spacialCoord);
      vec3 velocity = currSpacial.rgb;

      vec3 newPos = position+velocity*.05;
      vec3 newFloorPos = floor(newPos);
      if (any(notEqual(floorPos, newFloorPos))) {  // Moving into new cell
        vec2 newSpacialCoord = posToSpacial(newFloorPos);
        if (texture2D(spacial, newSpacialCoord).a == -1.) {  // Cell is empty.
          bool canMoveIn = true;
          for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
              for (int z = -1; z <= 1; z++) {
                if (canMoveIn && !(x == 0 && y == 0 && z == 0)) {
                  vec3 offset = vec3(x, y, z);
                  vec3 neighborPos = newFloorPos+offset;
                  if (all(greaterThan(neighborPos, vec3(-1))) && all(lessThan(neighborPos, worldSize))) {
                    vec4 neighbor = worldToSpacial(neighborPos);
                    float neighborIdx = neighbor.a;
                    if (neighborIdx != -1.) {
                      vec3 exactNeighborPos = spacialToPos(neighbor);
                      vec3 neighborVel = neighbor.rgb;
                      vec3 newNeighborPos = exactNeighborPos+neighborVel*.05;
                      vec3 newNeighborFloorPos = floor(newNeighborPos);
                      if (all(equal(newFloorPos, newNeighborFloorPos))) {
                        if (neighborIdx < currSpacial.a) {
                          gl_FragColor = vec4(position, 0);
                          canMoveIn = false;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          if (canMoveIn) {
            gl_FragColor = vec4(newPos, 0);
          }
        } else {  // New cell is currently occupied. Stay put.
          gl_FragColor = vec4(position, 0);
        }
      } else {  // Staying in same cell
        gl_FragColor = vec4(newPos, 0);
      }
    }`
  ),
  array: [],
  texture: null,
  processor: null
};

let spacialParams = {
  prgm: new Program(
    `precision mediump float;

    attribute vec2 pos;
    varying vec2 texPos;

    void main() {
      texPos = .5*(1.+pos);
      gl_Position = vec4(pos, 0, 1);
    }`,
    `precision mediump float;

    uniform sampler2D spacial;
    uniform sampler2D position;
    varying vec2 texPos;

    uniform vec3 worldSize;
    uniform vec2 spacialDim;
    uniform vec2 positionDim;

    vec3 spacialToWorld() {
      vec2 coord = floor(texPos*spacialDim);
      float idx = coord.y*spacialDim.x+coord.x;

      float yz = worldSize.y*worldSize.z;
      float x = floor(idx/yz);
      float y = floor(mod(idx, yz)/worldSize.z);
      float z = mod(idx, worldSize.z);

      return vec3(x, y, z);
    }

    vec4 worldToSpacial(vec3 pos) {
      float idx = pos.x*worldSize.y*worldSize.z+pos.y*worldSize.z+pos.z;
      return texture2D(spacial, vec2(mod(idx, spacialDim.x)+.5, floor(idx/spacialDim.x)+.5)/spacialDim);
    }

    vec3 spacialToPos(vec4 spcial) {
      float idx = spcial.a;
      return texture2D(position, vec2(mod(idx, positionDim.x)+.5, floor(idx/positionDim.x)+.5)/positionDim).rgb;
    }

    void checkMovedIn() {
      vec3 worldPos = spacialToWorld();
      vec4 col = vec4(-1);
      bool cont = true;
      for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
          for (int z = -1; z <= 1; z++) {
            if (cont && !(x == 0 && y == 0 && z == 0)) {
              vec3 offset = vec3(x, y, z);
              vec3 neighborPos = worldPos+offset;
              if (all(greaterThan(neighborPos, vec3(-1))) && all(lessThan(neighborPos, worldSize))) {
                vec4 neighbor = worldToSpacial(neighborPos);
                if (neighbor.a != -1.) {
                  vec3 exactPos = spacialToPos(neighbor);
                  if (all(equal(worldPos, floor(exactPos)))) {
                    col = neighbor;
                    cont = false;
                  }
                }
              }
            }
          }
        }
      }
      gl_FragColor = col;
    }

    void checkColliding(vec3 exactPos, vec3 velocity, float posIdx) {
      vec3 worldPos = spacialToWorld();
      vec3 finalVel = velocity;
      bool cont = true;
      if (exactPos.x < .5 && finalVel.x < 0.) {
        finalVel.x = abs(finalVel.x);
      } else if (exactPos.x > worldSize.x-.5 && finalVel.x > 0.) {
        finalVel.x = -finalVel.x;
      } else if (exactPos.y < .5 && finalVel.y < 0.) {
        finalVel.y = abs(finalVel.y);
      } else if (exactPos.y > worldSize.y-.5 && finalVel.y > 0.) {
        finalVel.y = -finalVel.y;
      } else if (exactPos.z < .5 && finalVel.z < 0.) {
        finalVel.z = abs(finalVel.z);
      } else if (exactPos.z > worldSize.z-.5 && finalVel.z > 0.) {
        finalVel.z = -finalVel.z;
      }
      for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
          for (int z = -1; z <= 1; z++) {
            if (cont && !(x == 0 && y == 0 && z == 0)) {
              vec3 offset = vec3(x, y, z);
              vec3 neighborPos = worldPos+offset;
              if (all(greaterThan(neighborPos, vec3(-1))) && all(lessThan(neighborPos, worldSize))) {
                vec4 neighbor = worldToSpacial(neighborPos);
                if (neighbor.a >= 0.) {  // Particle exists in neighboring cell
                  vec3 exactNeighborPos = spacialToPos(neighbor);
                  vec3 neighborVel = neighbor.rgb;
                  vec3 dPos = exactPos-exactNeighborPos;
                  vec3 dVel = neighborVel-finalVel;
                  if (dot(dPos, dVel) > 0.) {
                    float dist = length(dPos);
                    if (dist <= 1.1) {
                      vec3 normal = dPos/dist;
                      float dSpeed = dot(normal, dVel);
                      finalVel += dSpeed*normal;
                      // cont = false;
                    }
                  }
                }
              }
            }
          }
        }
      }
      gl_FragColor = vec4(finalVel, posIdx);
    }

    void main() {
      vec4 current = texture2D(spacial, texPos);

      if (current.a == -1.) {
        checkMovedIn();
      } else {
        vec3 approxPos = spacialToWorld();
        vec3 exactPos = spacialToPos(current);
        if (all(equal(approxPos, floor(exactPos)))) {
          checkColliding(exactPos, current.rgb, current.a);
        } else {
          checkMovedIn();
        }
      }
    }`
  ),
  array: Array(CELLS*4).fill(-1),
  texture: null,
  processor: null
};

function pos1Dto3D(idx) {
  let x = Math.floor(idx/(WORLD_SIZE[1]*WORLD_SIZE[2]));
  let y = Math.floor(idx%(WORLD_SIZE[1]*WORLD_SIZE[2])/WORLD_SIZE[2]);
  let z = idx%WORLD_SIZE[2];
  return [x, y, z];
}

function setupProcessors() {
  let possibleIndices = [...Array(CELLS).keys()];
  for (let i = 0; i < PARTICLES; i++) {
    let index = possibleIndices.splice(Math.floor(Math.random()*(CELLS-i)), 1)[0];
    let pos = pos1Dto3D(index);
    pos = pos.map((n, i) => {
      if (n == 0) return n+Math.random()/2+.5;
      else if (n+1 == WORLD_SIZE[i]) return n+Math.random()/2;
      return n+Math.random();
    });
    positionParams.array.push(...pos, 0);

    let velocity = [Math.random(), Math.random(), Math.random()].minus(.5).normalized();
    spacialArrayIdx = index*4;
    spacialParams.array[spacialArrayIdx] = velocity[0];
    spacialParams.array[spacialArrayIdx+1] = velocity[1];
    spacialParams.array[spacialArrayIdx+2] = velocity[2];
    spacialParams.array[spacialArrayIdx+3] = i;
  }
  positionParams.texture = dataTexResolveSize(positionParams.array);
  positionParams.processor = new ParProc(positionParams.prgm);

  spacialParams.texture = dataTexResolveSize(spacialParams.array);
  spacialParams.processor = new ParProc(spacialParams.prgm);

  positionParams.processor.setUniform("worldSize", WORLD_SIZE, "uniform3fv");
  positionParams.processor.setUniform("spacialDim", [spacialParams.texture.width, spacialParams.texture.height], "uniform2fv");
  positionParams.processor.setUniform("positionDim", [positionParams.texture.width, positionParams.texture.height], "uniform2fv");

  spacialParams.processor.setUniform("worldSize", WORLD_SIZE, "uniform3fv");
  spacialParams.processor.setUniform("spacialDim", [spacialParams.texture.width, spacialParams.texture.height], "uniform2fv");
  spacialParams.processor.setUniform("positionDim", [positionParams.texture.width, positionParams.texture.height], "uniform2fv");

  if (spheresOrPoints) {
    sphere.setFloat(0, positionParams.texture.width);
    sphere.setFloat(1, positionParams.texture.height);
  } else {
    GLOBAL_GL.uniform2f(pointPrgm.uniformLocationOf("texDim"), positionParams.texture.width, positionParams.texture.height);
  }
}

function runProcessors() {
  spacialParams.texture.use(positionParams.prgm, "spacial", 2);
  positionParams.processor.run({
    texIn: positionParams.texture,
    texInName: "position"
  });

  if (spheresOrPoints) {
    positionParams.texture.use(sphere.prgm, "dataTex", 4);
    sphere.draw();
  } else {
    positionParams.texture.use(pointPrgm, "dataTex", 4);
    pointVertices.draw();
  }

  positionParams.texture.use(spacialParams.prgm, "position", 2);
  spacialParams.processor.run({
    texIn: spacialParams.texture,
    texInName: "spacial"
  });
}

setupProcessors();
function draw() {
  runProcessors();
}
