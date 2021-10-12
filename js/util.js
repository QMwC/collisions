// Documentation: https://quantum.georgetown.domains/3D-Tutorial

window.PARAMETRIC_BUFFER = undefined;
window.GLOBAL_CAMERA = undefined;
window.GLOBAL_GL = undefined;
window.GLOBAL_CANVAS = undefined;
window.GLOBAL_PROGRAMS = {};
window.GLOBAL_TRANSFORMATIONS = [[[1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1]]];
window.GLOBAL_ARROW = undefined;
window.GLOBAL_INTERVALS = {};
window._ = undefined;

function pushMatrix() {
  if (GLOBAL_TRANSFORMATIONS.length < 10000) {
    GLOBAL_TRANSFORMATIONS.push(GLOBAL_TRANSFORMATIONS[GLOBAL_TRANSFORMATIONS.length-1]);
  } else {
    alert("Cannot push any more transformations matrices. Please remember to pop pushed matrices.");
    throw "Cannot push any more transformations matrices. Please remember to pop pushed matrices.";
  }
}

function popMatrix() {
  if (GLOBAL_TRANSFORMATIONS.length === 1) {
    GLOBAL_TRANSFORMATIONS[0] = [[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]];
  } else {
    GLOBAL_TRANSFORMATIONS.pop();
  }
}

function rotate(x=0, y=0, z=0) {
  let l = GLOBAL_TRANSFORMATIONS.length;
  let rotation = [
    [1, 0, 0, 0],
    [0, Math.cos(x), -Math.sin(x), 0],
    [0, Math.sin(x), Math.cos(x), 0],
    [0, 0, 0, 1]
  ].matmul([
    [Math.cos(y), 0, Math.sin(y), 0],
    [0, 1, 0, 0],
    [-Math.sin(y), 0, Math.cos(y), 0],
    [0, 0, 0, 1]
  ]).matmul([
    [Math.cos(z), -Math.sin(z), 0, 0],
    [Math.sin(z), Math.cos(z), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]);
  GLOBAL_TRANSFORMATIONS[l-1] = rotation.matmul(GLOBAL_TRANSFORMATIONS[l-1]);
}

function translate(x=0, y=0, z=0) {
  let l = GLOBAL_TRANSFORMATIONS.length;
  let translation = [
    [1, 0, 0, x],
    [0, 1, 0, y],
    [0, 0, 1, z],
    [0, 0, 0, 1]
  ];
  GLOBAL_TRANSFORMATIONS[l-1] = translation.matmul(GLOBAL_TRANSFORMATIONS[l-1]);
}

function scale(x=1, y=1, z=1) {
  let l = GLOBAL_TRANSFORMATIONS.length;
  let scaling = [
    [x, 0, 0, 0],
    [0, y, 0, 0],
    [0, 0, z, 0],
    [0, 0, 0, 1]
  ];
  GLOBAL_TRANSFORMATIONS[l-1] = scaling.matmul(GLOBAL_TRANSFORMATIONS[l-1]);
}

class Canvas {
  constructor(canvas, viewportMult=() => Math.min(innerWidth, innerHeight)<500?2:1) {
    GLOBAL_CANVAS = this.canvas = canvas;
    this.canvas.viewportMult = viewportMult;
    GLOBAL_GL = this.gl = canvas.getContext("webgl");

    if (!GLOBAL_GL) {
      console.log("WebGL not supported. Falling back on experimental-webgl.");
      GLOBAL_GL = canvas.getContext("experimental-webgl");
      if (!GLOBAL_GL) {
        alert("Your browser does not support WebGL.");
        throw "Your browser does not support WebGL.";
      }
    }

    GLOBAL_GL.getExtension("OES_texture_float");
    GLOBAL_GL.getExtension("WEBGL_color_buffer_float");
    GLOBAL_GL.enable(GLOBAL_GL.DEPTH_TEST);
    GLOBAL_GL.clearColor(0, 0, 0, 1);
    GLOBAL_GL.clear(GLOBAL_GL.COLOR_BUFFER_BIT | GLOBAL_GL.DEPTH_BUFFER_BIT);

    new ResizeObserver(() => this.resize()).observe(this.canvas);
    this.resize();
  }

  resize() {
    let style = getComputedStyle(this.canvas);
    let [width, height] = [Math.floor(Number(style.width.slice(0, -2))), Math.floor(Number(style.height.slice(0, -2)))].times(this.canvas.viewportMult());
    [this.canvas.width, this.canvas.height] = [width, height];
    GLOBAL_GL.viewport(0, 0, width, height);
    try {
      GLOBAL_CAMERA.sendToShaders(["coordMult"]);
      GLOBAL_CAMERA.updateText3D();
    } catch (e) { }
  }
}

class Program {
  constructor(vsSource, fsSource) {
    this.vsSource = vsSource;
    this.fsSource = fsSource;

    this.attribLocCache = {};  // {name0: location0, name1: location1}
    this.uniformLocCache = {};

    this.update();
    if (["matrix", "camera", "f", "coordMult"].filter(e => !this.vsSource.includes(e))) {
      GLOBAL_CAMERA.programs.push(this);
      GLOBAL_CAMERA.update(_, _, _, true);
    }
  }

  use() {
    GLOBAL_GL.useProgram(this.prgm);
  }

  clearAttributes() {
    Object.values(this.attribLocCache).forEach(loc => GLOBAL_GL.disableVertexAttribArray(loc));
  }

  clearAttribLocCache() {
    this.attribLocCache = {};
  }

  attribLocationOf(name) {
    GLOBAL_GL.useProgram(this.prgm);
    if (!(name in this.attribLocCache)) {
      this.attribLocCache[name] = GLOBAL_GL.getAttribLocation(this.prgm, name);
    }
    return this.attribLocCache[name];
  }

  uniformLocationOf(name) {
    GLOBAL_GL.useProgram(this.prgm);
    if (!(name in this.attribLocCache)) {
      this.uniformLocCache[name] = GLOBAL_GL.getUniformLocation(this.prgm, name);
    }
    return this.uniformLocCache[name];
  }

  update(vsSource = this.vsSource, fsSource = this.fsSource) {
    try {
      GLOBAL_GL.deleteProgram(this.prgm);
    } catch (e) { }

    this.vsSource = vsSource;
    this.fsSource = fsSource;

    this.vs = GLOBAL_GL.createShader(GLOBAL_GL.VERTEX_SHADER);
    this.fs = GLOBAL_GL.createShader(GLOBAL_GL.FRAGMENT_SHADER);
    GLOBAL_GL.shaderSource(this.vs, vsSource);
    GLOBAL_GL.shaderSource(this.fs, fsSource);
    GLOBAL_GL.compileShader(this.vs);
    if (!GLOBAL_GL.getShaderParameter(this.vs, GLOBAL_GL.COMPILE_STATUS)) {
      let infoLog = GLOBAL_GL.getShaderInfoLog(this.vs);
      try {
        let line = Number(infoLog.match(/\d+:\d+/)[0].split(":").pop());
        alert(infoLog+"\n"+this.vsSource.split("\n").map((l, i) => i+1+". "+l).slice(Math.max(line-2, 0), line+1).join("\n"));
      } catch (e) {
        alert("One of your GLSL equations gets a runtime error:"+"\n"+infoLog);
      }
  	}
    GLOBAL_GL.compileShader(this.fs);
    if (!GLOBAL_GL.getShaderParameter(this.fs, GLOBAL_GL.COMPILE_STATUS)) {
      let infoLog = GLOBAL_GL.getShaderInfoLog(this.fs);
      try {
        let line = Number(infoLog.match(/\d+:\d+/)[0].split(":").pop());
        alert(infoLog+"\n"+this.fsSource.split("\n").map((l, i) => i+1+". "+l).slice(Math.max(line-2, 0), line+1).join("\n"));
      } catch (e) {
        alert("One of your GLSL equations gets a runtime error:"+"\n"+infoLog);
      }
  	}
    this.prgm = GLOBAL_GL.createProgram();
    GLOBAL_GL.attachShader(this.prgm, this.vs);
    GLOBAL_GL.attachShader(this.prgm, this.fs);
    GLOBAL_GL.linkProgram(this.prgm);
    this.use();
  }
}

class Vertices {
  constructor(array, bufferType=GLOBAL_GL.ARRAY_BUFFER, drawOption=GLOBAL_GL.STATIC_DRAW) {
    this.array = array;
    this.bufferType = bufferType;
    this.drawOption = drawOption;

    this.defaultDrawMode = GLOBAL_GL.TRIANGLES;
    this.updateBuffer();
  }

  updateBuffer(array=this.array, bufferType=this.bufferType, drawOption=this.drawOption) {
    try {
      GLOBAL_GL.deleteBuffer(this.buffer);
    } catch (e) { }

    if (typeof(array) === "number") {  // Indicates number of parametric vertices
      this.buffer = PARAMETRIC_BUFFER;
    } else {
      this.buffer = GLOBAL_GL.createBuffer();
      GLOBAL_GL.bindBuffer(bufferType, this.buffer);
      GLOBAL_GL.bufferData(bufferType, array, drawOption);
    }
    GLOBAL_GL.bindBuffer(GLOBAL_GL.ARRAY_BUFFER, null);
  }

  addAttributes(prgm, attribNames, sizes, types, offsets, strides=Array(sizes.length).fill(sizes.sum()*4)) {
    this.prgm = prgm;
    this.attribNames = attribNames;
    this.sizes = sizes;
    this.types = types;
    this.strides = strides;
    this.offsets = offsets;

    prgm.use();
    prgm.clearAttributes();

    let totalSize = 0;
    GLOBAL_GL.bindBuffer(this.bufferType, this.buffer);
    attribNames.forEach((attribName, i) => {
      let attribLoc = prgm.attribLocationOf(attribName);
      GLOBAL_GL.vertexAttribPointer(
        attribLoc,
        sizes[i],
        types[i],
        false,
        strides[i],
        offsets[i]
      );
      GLOBAL_GL.enableVertexAttribArray(attribLoc);

      totalSize += sizes[i];
    });
    GLOBAL_GL.bindBuffer(GLOBAL_GL.ARRAY_BUFFER, null);
    this.count = (this.array.length||this.array)/totalSize;
  }

  use(prgm=this.prgm) {
    this.addAttributes(prgm, this.attribNames, this.sizes, this.types, this.offsets, this.strides);
  }

  draw(prgm=this.prgm, mode=this.defaultDrawMode, first=0, count=this.count) {
    this.use(prgm);
    let l = GLOBAL_TRANSFORMATIONS.length;
    if (this.computeParams) {  // Is a Parametric1D
      this.computeParams.dataTexture.use(prgm, "tex", 0);
    } else if (this.added) {  // Is a Parametric2D
      for (let i in this.added) {
        let p1d = this.added[i];
        GLOBAL_GL.uniform2f(
          prgm.uniformLocationOf("texDim"+i),
          p1d.computeParams.dataTexture.width,
          p1d.computeParams.dataTexture.height
        );
        GLOBAL_GL.uniform3f(
          prgm.uniformLocationOf("rBounds"+i),
          p1d.t0,
          p1d.t1,
          p1d.dt
        );
        p1d.computeParams.dataTexture.use(prgm, "tex"+i, Number(i));
      }
    }
    GLOBAL_GL.uniformMatrix4fv(prgm.uniformLocationOf("transformation"), false, GLOBAL_TRANSFORMATIONS[l-1].transpose().flat());
    GLOBAL_GL.drawArrays(mode, first, count);
  }
}

class ComputeVertices extends Vertices {
  constructor() {
    super(new Float32Array([
      -1, -1,
      -1, 1,
      1, -1,
      1, 1
    ]));
    this.defaultDrawMode = GLOBAL_GL.TRIANGLE_STRIP;
  }
}

class Parametric1D extends Vertices {
  constructor(r, t0=0, t1=2*Math.PI, dt=.1, lineRadius=.05, segments=Math.max(25, Math.round(lineRadius*50)), color=[1, 0, .5]) {
    if (!PARAMETRIC_BUFFER) {
      PARAMETRIC_BUFFER = GLOBAL_GL.createBuffer();
      GLOBAL_GL.bindBuffer(GLOBAL_GL.ARRAY_BUFFER, PARAMETRIC_BUFFER);
      GLOBAL_GL.bufferData(GLOBAL_GL.ARRAY_BUFFER, new Float32Array(Array(1e7).keys()), GLOBAL_GL.STATIC_DRAW);
    }
    if (!("sum" in GLOBAL_PROGRAMS)) {
      GLOBAL_PROGRAMS.sum = new Program(
        `precision mediump float;

        attribute vec2 pos;
        varying vec2 texPos;

        void main() {
          texPos = .5*(1.+pos);
          gl_Position = vec4(pos, 0, 1);
        }`,
        `precision mediump float;

        uniform sampler2D tex;
        varying vec2 texPos;
        uniform float steps, stepSize, startIdx, endIdx, size;
        uniform vec4 sumOver;

        void main() {
          float sum = 0.;
          for (float x = 0.; x < 3.5; x += 1.) {
            for (float y = 0.; y < 3.5; y += 1.) {
              vec2 pos = texPos+(vec2(x, y)-.5*(steps-1.))*stepSize;
              vec2 cornerPos = pos-.5*stepSize;
              float idx = floor(size*(cornerPos.y+cornerPos.x*stepSize)+.5);
              if (idx >= startIdx && idx < endIdx && x < steps-.5 && y < steps-.5) {  // For some reason, <= endIdx sometimes includes 1 too many
                sum += dot(texture2D(tex, pos), sumOver);
              }
            }
          }
          gl_FragColor = sum*sumOver;
        }`
      );
    }
    super((segments+1)*2*Math.round((t1-t0)/dt));

    this.defaultDrawMode = GLOBAL_GL.TRIANGLE_STRIP;

    this.r = r;
    this.t0 = t0;
    this.t1 = t1;
    this.dt = dt;
    this.lineRadius = lineRadius;
    this.segments = segments;
    this.color = color;

    this.vsSource = `precision mediump float;

      attribute float idx;

      uniform sampler2D tex;

      uniform float t0, t1, dt;

      uniform float segments;
      uniform float lineRadius;
      uniform vec2 texDim;

      uniform vec2 coordMult;
      uniform mat3 matrix;
      uniform vec3 camera;
      uniform float near, far;

      uniform mat4 transformation;

      varying vec3 vNormal, vPos, vCam;

      vec3 rRow(float row) {
        return texture2D(tex, (vec2(mod(row, texDim.x), floor(row/texDim.x))+.5)/texDim).rgb;
      }

      vec3 r(float t) {
        float row = floor((t-t0)/dt+.5);
        return rRow(row);
      }

      vec4 proj3Dto2D(vec3 v) {
        vec3 new = (v-camera)*matrix;
        return vec4(new.xy*coordMult, 2.*(1.-new.z/near)/(1./far-1./near)-new.z, new.z);
      }

      float radius(float t) {
        return lineRadius;
      }

      void main() {
        float idxForRounding = idx+.5;
        float verticesPerRow = (segments+1.)*2.;
        float row = floor(idxForRounding/verticesPerRow)+floor(mod(idxForRounding, 2.));
        float rowVertexIdx = floor(mod(idxForRounding, verticesPerRow));
        float segment = floor(.5*rowVertexIdx+.1);

        vec3 point = rRow(row);

        vec3 parallel = normalize(rRow(row+1.)-point);

        vec3 normal1 = normalize(vec3(parallel.zz, -parallel.x-parallel.y));
        vec3 normal2 = cross(parallel, normal1);
        float angle = 6.2831853072*segment/segments;
        vec3 normal = normal1*cos(angle)+normal2*sin(angle);
        vec3 posOnLine = (transformation*vec4(point, 1)).xyz;
        vPos = (transformation*vec4(point+lineRadius*normal, 1)).xyz;
        vNormal = vPos-posOnLine;
        vCam = camera;

        gl_Position = proj3Dto2D(vPos);
        //mainend
      }`;
    this.fsSource = `precision mediump float;

      uniform vec3 col;
      uniform bool shadow;

      varying vec3 vNormal, vPos, vCam;

      void main() {
        float ambient = .2;

        if (shadow) {
          gl_FragColor = vec4(ambient*col, 1);
        } else {
          vec3 lightDir1 = vec3(0, 0, -1);
          vec3 lightDir2 = vec3(-1, 0, 0);

          vec3 normal = normalize(vNormal);

          float diffuseStrength = .3;
          float diffuse = diffuseStrength*(max(0., dot(lightDir1, -normal))+max(0., dot(lightDir2, -normal)));

          float specularStrength = .5;
          vec3 viewDir = normalize(vCam-vPos);
          vec3 reflectDir1 = reflect(lightDir1, normal);
          vec3 reflectDir2 = reflect(lightDir2, normal);
          float specular = specularStrength*(pow(max(0., dot(viewDir, reflectDir1)), 16.)+pow(max(0., dot(viewDir, reflectDir2)), 16.));

          vec3 result = (ambient+diffuse+specular)*col;
          gl_FragColor = vec4(result, 1);
        }
      }`;
    this.prgm = new Program(this.vsSource, this.fsSource);

    this.addAttributes(
      this.prgm,
      ["idx"],
      [1],
      [GLOBAL_GL.FLOAT],
      [0]
    );

    this.additionalCalculations = 50;
    let dataTextureSize = 256;
    let tCount = Math.round((t1-t0)/dt)+this.additionalCalculations;
    if (tCount > dataTextureSize**2) {
      dataTextureSize = 2**Math.ceil(Math.log2(Math.sqrt(tCount)));
    }

    this.computeParams = {
      dataTexture: new DataTexture(_, dataTextureSize, dataTextureSize, 15),
      added: {},
      linked: {},
      framebuffer: GLOBAL_GL.createFramebuffer(),
      vertices: new ComputeVertices,
      prgm: new Program(
        `precision mediump float;

        attribute vec2 pos;
        varying vec2 texPos;
        uniform vec2 texPosMult;

        void main() {
          texPos = .5*(1.+pos)*texPosMult;
          gl_Position = vec4(pos, 0, 1);
        }`,
        `precision mediump float;

        const float PI = 3.141592653589793;

        varying vec2 texPos;

        uniform float t0, t1, dt;
        uniform vec2 dim;

        uniform float float0, float1, float2, float3, float4, float5, float6, float7, float8, float9, float10, float11, float12, float13, float14, float15;
        uniform vec4 vector0, vector1, vector2, vector3, vector4, vector5, vector6, vector7, vector8, vector9, vector10, vector11, vector12, vector13, vector14, vector15;
        uniform mat4 matrix0, matrix1, matrix2, matrix3, matrix4, matrix5, matrix6, matrix7, matrix8, matrix9, matrix10, matrix11, matrix12, matrix13, matrix14, matrix15;

        uniform sampler2D tex0, tex1, tex2, tex3, tex4, tex5, tex6, tex7, tex8, tex9, tex10, tex11, tex12, tex13, tex14, tex15;
      uniform vec2 texDim0, texDim1, texDim2, texDim3, texDim4, texDim5, texDim6, texDim7, texDim8, texDim9, texDim10, texDim11, texDim12, texDim13, texDim14, texDim15;
      uniform vec3 rBounds0, rBounds1, rBounds2, rBounds3, rBounds4, rBounds5, rBounds6, rBounds7, rBounds8, rBounds9, rBounds10, rBounds11, rBounds12, rBounds13, rBounds14, rBounds15;

        vec3 r0(float t) {
          float texIdx = floor((t-rBounds0.x)/rBounds0.z+.5);
          return texture2D(tex0, vec2(mod(texIdx, texDim0.x)+.5, floor(texIdx/texDim0.x)+.5)/texDim0).rgb;
        }

        vec3 r1(float t) {
          float texIdx = floor((t-rBounds1.x)/rBounds1.z+.5);
          return texture2D(tex1, vec2(mod(texIdx, texDim1.x)+.5, floor(texIdx/texDim1.x)+.5)/texDim1).rgb;
        }

        vec3 r2(float t) {
          float texIdx = floor((t-rBounds2.x)/rBounds2.z+.5);
          return texture2D(tex2, vec2(mod(texIdx, texDim2.x)+.5, floor(texIdx/texDim2.x)+.5)/texDim2).rgb;
        }

        vec3 r3(float t) {
          float texIdx = floor((t-rBounds3.x)/rBounds3.z+.5);
          return texture2D(tex3, vec2(mod(texIdx, texDim3.x)+.5, floor(texIdx/texDim3.x)+.5)/texDim3).rgb;
        }

        vec3 r4(float t) {
          float texIdx = floor((t-rBounds4.x)/rBounds4.z+.5);
          return texture2D(tex4, vec2(mod(texIdx, texDim4.x)+.5, floor(texIdx/texDim4.x)+.5)/texDim4).rgb;
        }

        vec3 r5(float t) {
          float texIdx = floor((t-rBounds5.x)/rBounds5.z+.5);
          return texture2D(tex5, vec2(mod(texIdx, texDim5.x)+.5, floor(texIdx/texDim5.x)+.5)/texDim5).rgb;
        }

        vec3 r6(float t) {
          float texIdx = floor((t-rBounds6.x)/rBounds6.z+.5);
          return texture2D(tex6, vec2(mod(texIdx, texDim6.x)+.5, floor(texIdx/texDim6.x)+.5)/texDim6).rgb;
        }

        vec3 r7(float t) {
          float texIdx = floor((t-rBounds7.x)/rBounds7.z+.5);
          return texture2D(tex7, vec2(mod(texIdx, texDim7.x)+.5, floor(texIdx/texDim7.x)+.5)/texDim7).rgb;
        }

        vec3 r8(float t) {
          float texIdx = floor((t-rBounds8.x)/rBounds8.z+.5);
          return texture2D(tex8, vec2(mod(texIdx, texDim8.x)+.5, floor(texIdx/texDim8.x)+.5)/texDim8).rgb;
        }

        vec3 r9(float t) {
          float texIdx = floor((t-rBounds9.x)/rBounds9.z+.5);
          return texture2D(tex9, vec2(mod(texIdx, texDim9.x)+.5, floor(texIdx/texDim9.x)+.5)/texDim9).rgb;
        }

        vec3 r10(float t) {
          float texIdx = floor((t-rBounds10.x)/rBounds10.z+.5);
          return texture2D(tex10, vec2(mod(texIdx, texDim10.x)+.5, floor(texIdx/texDim10.x)+.5)/texDim10).rgb;
        }

        vec3 r11(float t) {
          float texIdx = floor((t-rBounds11.x)/rBounds11.z+.5);
          return texture2D(tex11, vec2(mod(texIdx, texDim11.x)+.5, floor(texIdx/texDim11.x)+.5)/texDim11).rgb;
        }

        vec3 r12(float t) {
          float texIdx = floor((t-rBounds12.x)/rBounds12.z+.5);
          return texture2D(tex12, vec2(mod(texIdx, texDim12.x)+.5, floor(texIdx/texDim12.x)+.5)/texDim12).rgb;
        }

        vec3 r13(float t) {
          float texIdx = floor((t-rBounds13.x)/rBounds13.z+.5);
          return texture2D(tex13, vec2(mod(texIdx, texDim13.x)+.5, floor(texIdx/texDim13.x)+.5)/texDim13).rgb;
        }

        vec3 r14(float t) {
          float texIdx = floor((t-rBounds14.x)/rBounds14.z+.5);
          return texture2D(tex14, vec2(mod(texIdx, texDim14.x)+.5, floor(texIdx/texDim14.x)+.5)/texDim14).rgb;
        }

        vec3 r15(float t) {
          float texIdx = floor((t-rBounds15.x)/rBounds15.z+.5);
          return texture2D(tex15, vec2(mod(texIdx, texDim15.x)+.5, floor(texIdx/texDim15.x)+.5)/texDim15).rgb;
        }

        vec3 r(float t) {
          ${r}
        }

        float texPosToIdx() {
          vec2 texCoord = texPos*dim-.5;
          return texCoord.y*dim.x+texCoord.x;
        }

        void main() {
          float idx = texPosToIdx();
          gl_FragColor = vec4(r(t0+idx*dt), 0);
        }`
      )
    };
    this.computeParams.vertices.addAttributes(
      this.computeParams.prgm,
      ["pos"],
      [2],
      [GLOBAL_GL.FLOAT],
      [0]
    );

    this.sendToShader();
  }

  compute() {
    for (let i in this.computeParams.added) {
      let p1d = this.computeParams.added[i];
      GLOBAL_GL.uniform2f(
        this.computeParams.prgm.uniformLocationOf("texDim"+i),
        p1d.computeParams.dataTexture.width,
        p1d.computeParams.dataTexture.height
      );
      GLOBAL_GL.uniform3f(
        this.computeParams.prgm.uniformLocationOf("rBounds"+i),
        p1d.t0,
        p1d.t1,
        p1d.dt
      );
      p1d.computeParams.dataTexture.use(this.computeParams.prgm, "tex"+i, Number(i));
    }
    let points = Math.round((this.t1-this.t0)/this.dt+1)+this.additionalCalculations;
    let computeHeight = Math.ceil(points/this.computeParams.dataTexture.width);
    let computeWidth = this.computeParams.dataTexture.width;

    GLOBAL_GL.uniform2fv(this.computeParams.prgm.uniformLocationOf("texPosMult"), [computeWidth, computeHeight].over(this.computeParams.dataTexture.height));

    GLOBAL_GL.bindFramebuffer(GLOBAL_GL.FRAMEBUFFER, this.computeParams.framebuffer);
    GLOBAL_GL.framebufferTexture2D(GLOBAL_GL.FRAMEBUFFER, GLOBAL_GL.COLOR_ATTACHMENT0, GLOBAL_GL.TEXTURE_2D, this.computeParams.dataTexture.texture, 0);
    GLOBAL_GL.viewport(0, 0, computeWidth, computeHeight);

    this.computeParams.vertices.draw();

    GLOBAL_GL.viewport(0, 0, GLOBAL_CANVAS.width, GLOBAL_CANVAS.height);
    GLOBAL_GL.bindFramebuffer(GLOBAL_GL.FRAMEBUFFER, null);
  }

  drawShadow(direction="X", offset=0, color=[0, 0, 0]) {
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("shadow"), 1);
    GLOBAL_GL.uniform3fv(this.prgm.uniformLocationOf("col"), color);
    let scaling = ["x", "y", "z"].map(l => 1-(l === direction.toLowerCase()));
    let translation = scaling.map(n => offset*(1-n));
    pushMatrix();
    scale(...scaling);
    translate(...translation);
    this.draw();
    popMatrix();
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("shadow"), 0);
    GLOBAL_GL.uniform3fv(this.prgm.uniformLocationOf("col"), this.color);
  }

  sum(t0=this.t0, t1=this.t1, sumOver="X") {
    if (!this.computeParams.sumVertices) {
      this.computeParams.sumVertices = new ComputeVertices;
      this.computeParams.sumVertices.addAttributes(
        GLOBAL_PROGRAMS.sum,
        ["pos"],
        [2],
        [GLOBAL_GL.FLOAT],
        [0]
      );
    }

    let texIn = this.computeParams.dataTexture;

    if (texIn.width === texIn.height) {
      GLOBAL_GL.uniform1f(GLOBAL_PROGRAMS.sum.uniformLocationOf("startIdx"), Math.round((t0-this.t0)/this.dt));
      GLOBAL_GL.uniform1f(GLOBAL_PROGRAMS.sum.uniformLocationOf("endIdx"), Math.round((t1-this.t0)/this.dt));
      GLOBAL_GL.uniform1f(GLOBAL_PROGRAMS.sum.uniformLocationOf("dt"), this.dt);
      GLOBAL_GL.uniform4f(GLOBAL_PROGRAMS.sum.uniformLocationOf("sumOver"), "Xx".includes(sumOver), "Yy".includes(sumOver), "Zz".includes(sumOver), 0);

      let i = 0;
      while (true) {
        let steps = texIn.width%4 === 0?4:2;

        GLOBAL_GL.uniform1f(GLOBAL_PROGRAMS.sum.uniformLocationOf("steps"), steps);
        GLOBAL_GL.uniform1f(GLOBAL_PROGRAMS.sum.uniformLocationOf("stepSize"), 1/texIn.width);
        GLOBAL_GL.uniform1f(GLOBAL_PROGRAMS.sum.uniformLocationOf("size"), texIn.width**2);

        texIn.use(GLOBAL_PROGRAMS.sum, "tex", 14);
        let texOut = new DataTexture(_, texIn.width/steps, _, 15);

        GLOBAL_GL.bindFramebuffer(GLOBAL_GL.FRAMEBUFFER, this.computeParams.framebuffer);
        GLOBAL_GL.framebufferTexture2D(GLOBAL_GL.FRAMEBUFFER, GLOBAL_GL.COLOR_ATTACHMENT0, GLOBAL_GL.TEXTURE_2D, texOut.texture, 0);

        GLOBAL_GL.viewport(0, 0, texOut.width, texOut.width);
        this.computeParams.sumVertices.draw();

        if (texOut.width === 1) {
          let pixels = new Float32Array(4*texOut.width**2);
          GLOBAL_GL.readPixels(0, 0, texOut.width, texOut.width, GLOBAL_GL.RGBA, GLOBAL_GL.FLOAT, pixels);
          GLOBAL_GL.bindFramebuffer(GLOBAL_GL.FRAMEBUFFER, null);
          GLOBAL_GL.viewport(0, 0, GLOBAL_CANVAS.width, GLOBAL_CANVAS.height);
          texOut.del();
          return Array.from(pixels).sum();
        }

        GLOBAL_GL.uniform1f(GLOBAL_PROGRAMS.sum.uniformLocationOf("startIdx"), 0);

        if (i > 0) {
          texIn.del();
        };
        texIn = texOut;
        GLOBAL_GL.bindFramebuffer(GLOBAL_GL.FRAMEBUFFER, null);
        i++;
      }
    }
  }

  integral(t0=this.t0, t1=this.t1, body="x") {
    return this.sum(t0, t1, body[0])*this.dt;
  }

  update(t0=this.t0, t1=this.t1, dt=this.dt, lineRadius=this.lineRadius, segments=this.segments, color=this.color) {
    this.t0 = t0;
    this.t1 = t1;
    this.dt = dt;
    this.lineRadius = lineRadius;
    this.segments = segments;
    this.color = color;

    this.count = this.array = (segments+1)*2*Math.round((t1-t0)/dt);  // Really specifying array size/count
    this.sendToShader();

    let childrenToUpdate = [];
    for (let selfVar in this.computeParams.linked) {
    for (let child of this.computeParams.linked[selfVar]) {
        child.child[child.variable] = this[selfVar];
        if (!childrenToUpdate.includes(child.child)) {
          childrenToUpdate.push(child.child);
        }
      }
    }
    for (let child of childrenToUpdate) {
      child.update();
    }
  }

  setParametric1D(equationNumber, parametric1D, link={}) {
    let alreadyAdded = this.computeParams.added[equationNumber];
    if (alreadyAdded) {  // unlink previous
      for (let key in alreadyAdded.computeParams.linked) {
        for (let i = 0; i < alreadyAdded.computeParams.linked[key].length; i++) {
          if (alreadyAdded.computeParams.linked[key][i].child === this) {
            alreadyAdded.computeParams.linked[key].splice(i, i+1);
          }
        }
      }
    }
    this.computeParams.added[equationNumber] = parametric1D;
    for (let key in link) {
      if (!(key in parametric1D.computeParams.linked)) {
        parametric1D.computeParams.linked[key] = [];
      }
      parametric1D.computeParams.linked[key].push({child: this, variable: link[key]});
    }
    this.update();
  }

  setFloat(variableNumber, value) {
    if (variableNumber > 15) {
      alert("Highest variable number for setFloat is 15.");
      throw "Highest variable number for setFloat is 15.";
    }
    GLOBAL_GL.uniform1f(this.computeParams.prgm.uniformLocationOf("float"+variableNumber), value);
  }

  setVector(variableNumber, value) {
    if (variableNumber > 15) {
      alert("Highest variable number for setVector is 15.");
      throw "Highest variable number for setVector is 15.";
    }
    if (value.length < 4) {
      alert("Must supply 4-component array with setVector!");
      throw "Must supply 4-component arrray with setVector!";
    }
    GLOBAL_GL.uniform4fv(this.computeParams.prgm.uniformLocationOf("vector"+variableNumber), value);
  }

  setMatrix(variableNumber, value) {
    if (variableNumber > 15) {
      alert("Highest variable number for setMatrix is 15.");
      throw "Highest variable number for setMatrix is 15.";
    }
    if (value.length < 4) {
      alert("Must supply 4x4 array with setMatrix!");
      throw "Must supply 4x4 array with setMatrix!";
    }
    GLOBAL_GL.uniformMatrix4fv(this.computeParams.prgm.uniformLocationOf("matrix"+variableNumber), value.transpose().flat(1));
  }

  sendToShader() {
    GLOBAL_GL.uniform1f(this.computeParams.prgm.uniformLocationOf("t0"), this.t0);
    GLOBAL_GL.uniform1f(this.computeParams.prgm.uniformLocationOf("t1"), this.t1);
    GLOBAL_GL.uniform1f(this.computeParams.prgm.uniformLocationOf("dt"), this.dt);
    GLOBAL_GL.uniform2f(this.computeParams.prgm.uniformLocationOf("dim"), this.computeParams.dataTexture.width, this.computeParams.dataTexture.height);

    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("lineRadius"), this.lineRadius);
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("segments"), this.segments);
    GLOBAL_GL.uniform3fv(this.prgm.uniformLocationOf("col"), this.color);
    GLOBAL_GL.uniform2f(this.prgm.uniformLocationOf("texDim"), this.computeParams.dataTexture.width, this.computeParams.dataTexture.height);
  }
}

class Parametric2D extends Vertices {
  constructor(r, u0=0, u1=2*Math.PI, du=.1, v0=0, v1=2*Math.PI, dv=.1, color=[1, 0, 0]) {
    if (!PARAMETRIC_BUFFER) {
      PARAMETRIC_BUFFER = GLOBAL_GL.createBuffer();
      GLOBAL_GL.bindBuffer(GLOBAL_GL.ARRAY_BUFFER, PARAMETRIC_BUFFER);
      GLOBAL_GL.bufferData(GLOBAL_GL.ARRAY_BUFFER, new Float32Array(Array(1e7).keys()), GLOBAL_GL.STATIC_DRAW);
    }
    super(2*Math.round((u1-u0)/du+2)*Math.round((v1-v0)/dv));

    this.defaultDrawMode = GLOBAL_GL.TRIANGLE_STRIP;

    this.r = r;
    this.u0 = u0;
    this.u1 = u1;
    this.du = du;
    this.v0 = v0;
    this.v1 = v1;
    this.dv = dv;
    this.color = color;

    this.added = {};
    this.uCount = 2*Math.round((u1-u0)/du+2);
    this.vsSource = `precision mediump float;

      const float PI = 3.141592653589793;

      attribute float idx;

      uniform vec2 coordMult;
      uniform mat3 matrix;
      uniform vec3 camera;
      uniform float near, far;

      uniform float u0, u1, du, v0, v1, dv;

      uniform float uCount;

      uniform float float0, float1, float2, float3, float4, float5, float6, float7, float8, float9, float10, float11, float12, float13, float14, float15;
      uniform vec4 vector0, vector1, vector2, vector3, vector4, vector5, vector6, vector7, vector8, vector9, vector10, vector11, vector12, vector13, vector14, vector15;
      uniform mat4 matrix0, matrix1, matrix2, matrix3, matrix4, matrix5, matrix6, matrix7, matrix8, matrix9, matrix10, matrix11, matrix12, matrix13, matrix14, matrix15;

      varying vec3 vNormal, vPos, vCam;

      uniform sampler2D tex0, tex1, tex2, tex3, tex4, tex5, tex6, tex7, tex8, tex9, tex10, tex11, tex12, tex13, tex14, tex15;
      uniform vec2 texDim0, texDim1, texDim2, texDim3, texDim4, texDim5, texDim6, texDim7, texDim8, texDim9, texDim10, texDim11, texDim12, texDim13, texDim14, texDim15;
      uniform vec3 rBounds0, rBounds1, rBounds2, rBounds3, rBounds4, rBounds5, rBounds6, rBounds7, rBounds8, rBounds9, rBounds10, rBounds11, rBounds12, rBounds13, rBounds14, rBounds15;

      uniform mat4 transformation;

      vec3 r0(float t) {
        float texIdx = floor((t-rBounds0.x)/rBounds0.z+.5);
        return texture2D(tex0, vec2(mod(texIdx, texDim0.x)+.5, floor(texIdx/texDim0.x)+.5)/texDim0).rgb;
      }

      vec3 r1(float t) {
        float texIdx = floor((t-rBounds1.x)/rBounds1.z+.5);
        return texture2D(tex1, vec2(mod(texIdx, texDim1.x)+.5, floor(texIdx/texDim1.x)+.5)/texDim1).rgb;
      }

      vec3 r2(float t) {
        float texIdx = floor((t-rBounds2.x)/rBounds2.z+.5);
        return texture2D(tex2, vec2(mod(texIdx, texDim2.x)+.5, floor(texIdx/texDim2.x)+.5)/texDim2).rgb;
        }

      vec3 r3(float t) {
        float texIdx = floor((t-rBounds3.x)/rBounds3.z+.5);
        return texture2D(tex3, vec2(mod(texIdx, texDim3.x)+.5, floor(texIdx/texDim3.x)+.5)/texDim3).rgb;
      }

      vec3 r4(float t) {
        float texIdx = floor((t-rBounds4.x)/rBounds4.z+.5);
        return texture2D(tex4, vec2(mod(texIdx, texDim4.x)+.5, floor(texIdx/texDim4.x)+.5)/texDim4).rgb;
      }

      vec3 r5(float t) {
        float texIdx = floor((t-rBounds5.x)/rBounds5.z+.5);
        return texture2D(tex5, vec2(mod(texIdx, texDim5.x)+.5, floor(texIdx/texDim5.x)+.5)/texDim5).rgb;
      }

      vec3 r6(float t) {
        float texIdx = floor((t-rBounds6.x)/rBounds6.z+.5);
        return texture2D(tex6, vec2(mod(texIdx, texDim6.x)+.5, floor(texIdx/texDim6.x)+.5)/texDim6).rgb;
      }

      vec3 r7(float t) {
        float texIdx = floor((t-rBounds7.x)/rBounds7.z+.5);
        return texture2D(tex7, vec2(mod(texIdx, texDim7.x)+.5, floor(texIdx/texDim7.x)+.5)/texDim7).rgb;
      }

      vec3 r8(float t) {
        float texIdx = floor((t-rBounds8.x)/rBounds8.z+.5);
        return texture2D(tex8, vec2(mod(texIdx, texDim8.x)+.5, floor(texIdx/texDim8.x)+.5)/texDim8).rgb;
      }

      vec3 r9(float t) {
        float texIdx = floor((t-rBounds9.x)/rBounds9.z+.5);
        return texture2D(tex9, vec2(mod(texIdx, texDim9.x)+.5, floor(texIdx/texDim9.x)+.5)/texDim9).rgb;
      }

      vec3 r10(float t) {
        float texIdx = floor((t-rBounds10.x)/rBounds10.z+.5);
        return texture2D(tex10, vec2(mod(texIdx, texDim10.x)+.5, floor(texIdx/texDim10.x)+.5)/texDim10).rgb;
      }

      vec3 r11(float t) {
        float texIdx = floor((t-rBounds11.x)/rBounds11.z+.5);
        return texture2D(tex11, vec2(mod(texIdx, texDim11.x)+.5, floor(texIdx/texDim11.x)+.5)/texDim11).rgb;
      }

      vec3 r12(float t) {
        float texIdx = floor((t-rBounds12.x)/rBounds12.z+.5);
        return texture2D(tex12, vec2(mod(texIdx, texDim12.x)+.5, floor(texIdx/texDim12.x)+.5)/texDim12).rgb;
      }

      vec3 r13(float t) {
        float texIdx = floor((t-rBounds13.x)/rBounds13.z+.5);
        return texture2D(tex13, vec2(mod(texIdx, texDim13.x)+.5, floor(texIdx/texDim13.x)+.5)/texDim13).rgb;
      }

      vec3 r14(float t) {
        float texIdx = floor((t-rBounds14.x)/rBounds14.z+.5);
        return texture2D(tex14, vec2(mod(texIdx, texDim14.x)+.5, floor(texIdx/texDim14.x)+.5)/texDim14).rgb;
      }

      vec3 r15(float t) {
        float texIdx = floor((t-rBounds15.x)/rBounds15.z+.5);
        return texture2D(tex15, vec2(mod(texIdx, texDim15.x)+.5, floor(texIdx/texDim15.x)+.5)/texDim15).rgb;
      }

      vec3 r(float u, float v) {
        ${r}
      }

      vec3 transformedR(float u, float v) {
        return (transformation*vec4(r(u, v), 1)).xyz;
      }

      vec4 proj3Dto2D(vec3 v) {
        vec3 new = (v-camera)*matrix;
        return vec4(new.xy*coordMult, 2.*(1.-new.z/near)/(1./far-1./near)-new.z, new.z);
      }

      void main() {
        float idxForRounding = idx+.5;
        float bandIdx = floor(idxForRounding/uCount);
        float bandPointIdx = min(max(floor(mod(idxForRounding, uCount))-1., 0.), uCount-3.);
        float u = u0+du*floor(.5*bandPointIdx+.1);
        float v = v0+dv*(bandIdx+mod(bandPointIdx, 2.));
        float uNext = u+du;
        float vNext = v+dv;

        vPos = transformedR(u, v);
        vNormal = normalize(cross(transformedR(u, vNext)-vPos, transformedR(uNext, v)-vPos));
        vCam = camera;

        gl_Position = proj3Dto2D(vPos);
      }`;
    this.fsSource = `precision mediump float;

      uniform vec3 col;
      uniform bool shadow;

      varying vec3 vNormal, vPos, vCam;

      void main() {
        float ambient = .2;
        if (shadow) {
          gl_FragColor = vec4(ambient*col, 1);
        } else {
          vec3 lightDir1 = vec3(0, 0, -1);
          vec3 lightDir2 = vec3(-1, 0, 0);

          vec3 diff = vCam-vPos;

          vec3 normal = normalize(vNormal);
          if (dot(normal, vCam-vPos) < 0.) normal *= -1.;

          float diffuseStrength = .3;
          float diffuse = diffuseStrength*(max(0., dot(lightDir1, -normal))+max(0., dot(lightDir2, -normal)));

          float specularStrength = .5;
          vec3 viewDir = normalize(diff);
          vec3 reflectDir1 = reflect(lightDir1, normal);
          vec3 reflectDir2 = reflect(lightDir2, normal);
          float specular = specularStrength*(pow(max(0., dot(viewDir, reflectDir1)), 16.)+pow(max(0., dot(viewDir, reflectDir2)), 16.));

          vec3 result = (ambient+diffuse+specular)*col;
          gl_FragColor = vec4(result, 1);
        }
      }`;
    this.prgm = new Program(this.vsSource, this.fsSource);

    this.addAttributes(
      this.prgm,
      ["idx"],
      [1],
      [GLOBAL_GL.FLOAT],
      [0]
    );

    this.sendToShader();
  }

  compute() {}

  resolvePoles(ignore=[]) {
    this.shouldResolvePoles = true;

    let uSteps = Math.round((this.u1-this.u0)/this.du);
    let vSteps = Math.round((this.v1-this.v0)/this.dv);
    if (!ignore.includes("u0")) {this.u0 += .00001;}
    if (!ignore.includes("u1")) {this.u1 -= .00001;}
    if (!ignore.includes("v0")) {this.v0 += .00001;}
    if (!ignore.includes("v1")) {this.v1 -= .00001;}
    this.du = (this.u1-this.u0)/uSteps;
    this.dv = (this.v1-this.v0)/vSteps;

    this.uCount = 2*Math.round((this.u1-this.u0)/this.du+2);
    this.count = this.array = this.uCount*Math.round((this.v1-this.v0)/this.dv);  // Really specifying array size/count
    this.sendToShader();
  }

  update(u0=this.u0, u1=this.u1, du=this.du, v0=this.v0, v1=this.v1, dv=this.dv, color=this.color) {
    let ignore = [];
    if (this.shouldResolvePoles) {
      if (u0 === this.u0) {
        ignore.push("u0");
      } if (u1 === this.u1) {
        ignore.push("u1");
      } if (v0 === this.v0) {
        ignore.push("v0");
      } if (v1 === this.v1) {
        ignore.push("v1");
      }
    }
    this.u0 = u0;
    this.u1 = u1;
    this.du = du;
    this.v0 = v0;
    this.v1 = v1;
    this.dv = dv;
    this.color = color;
    if (this.shouldResolvePoles) {
      this.resolvePoles(ignore);
    } else {
      this.uCount = 2*Math.round((this.u1-this.u0)/this.du+2);
      this.count = this.array = this.uCount*Math.round((this.v1-this.v0)/this.dv);  // Really specifying array size/count
      this.sendToShader();
    }
  }

  drawShadow(direction="X", offset=0, color=[0, 0, 0]) {
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("shadow"), 1);
    GLOBAL_GL.uniform3fv(this.prgm.uniformLocationOf("col"), color);
    let scaling = ["x", "y", "z"].map(l => 1-(l === direction.toLowerCase()));
    let translation = scaling.map(n => offset*(1-n));
    pushMatrix();
    scale(...scaling);
    translate(...translation);
    this.draw();
    popMatrix();
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("shadow"), 0);
    GLOBAL_GL.uniform3fv(this.prgm.uniformLocationOf("col"), this.color);
  }

  setParametric1D(equationNumber, parametric1D, link={}) {
    let alreadyAdded = this.added[equationNumber];
    if (alreadyAdded) {  // unlink previous
      for (let key in alreadyAdded.computeParams.linked) {
        for (let i = 0; i < alreadyAdded.computeParams.linked[key].length; i++) {
          if (alreadyAdded.computeParams.linked[key][i].child === this) {
            alreadyAdded.computeParams.linked[key].splice(i, i+1);
          }
        }
      }
    }
    this.added[equationNumber] = parametric1D;
    for (let key in link) {
      if (!(key in parametric1D.computeParams.linked)) {
        parametric1D.computeParams.linked[key] = [];
      }
      parametric1D.computeParams.linked[key].push({child: this, variable: link[key]});
    }
  }

  setFloat(variableNumber, value) {
    if (variableNumber > 15) {
      alert("Highest variable number for setFloat is 15.");
      throw "Highest variable number for setFloat is 15.";
    }
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("float"+variableNumber), value);
  }

  setVector(variableNumber, value) {
    if (variableNumber > 15) {
      alert("Highest variable number for setVector is 15.");
      throw "Highest variable number for setVector is 15.";
    }
    if (value.length < 4) {
      alert("Must supply 4-component array with setVector!");
      throw "Must supply 4-component arrray with setVector!";
    }
    GLOBAL_GL.uniform4fv(this.prgm.uniformLocationOf("vector"+variableNumber), value);
  }

  setMatrix(variableNumber, value) {
    if (variableNumber > 15) {
      alert("Highest variable number for setMatrix is 15.");
      throw "Highest variable number for setMatrix is 15.";
    }
    if (value.length < 4) {
      alert("Must supply 4x4 array with setMatrix!");
      throw "Must supply 4x4 array with setMatrix!";
    }
    GLOBAL_GL.uniformMatrix4fv(this.prgm.uniformLocationOf("matrix"+variableNumber), value.transpose().flat(1));
  }

  sendToShader() {
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("u0"), this.u0);
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("u1"), this.u1);
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("du"), this.du);
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("v0"), this.v0);
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("v1"), this.v1);
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("dv"), this.dv);
    GLOBAL_GL.uniform3fv(this.prgm.uniformLocationOf("col"), this.color);
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("uCount"), this.uCount);
  }
}

class Texture {
  constructor(data, idx, width, height) {
    this.data = data;
    this.width = width;
    this.height = height;
    this.idx = idx;

    this.update();
  }

  update(data=this.data, idx=this.idx, width=this.width, height=this.height) {
    try {
      this.del();
    } catch (e) { }

    this.data = data;
    this.idx = idx;
    this.width = width;
    this.height = height;

    this.texture = GLOBAL_GL.createTexture();
    GLOBAL_GL.bindTexture(GLOBAL_GL.TEXTURE_2D, this.texture);
    if (data.length === undefined) {  // image
      if (data.complete) {
        GLOBAL_GL.texImage2D(GLOBAL_GL.TEXTURE_2D, 0, GLOBAL_GL.RGBA, GLOBAL_GL.RGBA, GLOBAL_GL.UNSIGNED_BYTE, data);
      } else {
        data.onload = () => {
          GLOBAL_GL.bindTexture(GLOBAL_GL.TEXTURE_2D, this.texture);
          GLOBAL_GL.texImage2D(GLOBAL_GL.TEXTURE_2D, 0, GLOBAL_GL.RGBA, GLOBAL_GL.RGBA, GLOBAL_GL.UNSIGNED_BYTE, data);
        };
      }
    } else {
      GLOBAL_GL.texImage2D(GLOBAL_GL.TEXTURE_2D, 0, GLOBAL_GL.RGBA, width, height, 0, GLOBAL_GL.RGBA, GLOBAL_GL.UNSIGNED_BYTE, data);
    }
    GLOBAL_GL.texParameteri(GLOBAL_GL.TEXTURE_2D, GLOBAL_GL.TEXTURE_WRAP_S, GLOBAL_GL.CLAMP_TO_EDGE);
    GLOBAL_GL.texParameteri(GLOBAL_GL.TEXTURE_2D, GLOBAL_GL.TEXTURE_WRAP_T, GLOBAL_GL.CLAMP_TO_EDGE);
    GLOBAL_GL.texParameteri(GLOBAL_GL.TEXTURE_2D, GLOBAL_GL.TEXTURE_MIN_FILTER, GLOBAL_GL.LINEAR);
    GLOBAL_GL.texParameteri(GLOBAL_GL.TEXTURE_2D, GLOBAL_GL.TEXTURE_MAG_FILTER, GLOBAL_GL.LINEAR);
  }

  use(prgm, name, idx=this.idx) {
    GLOBAL_GL.activeTexture(GLOBAL_GL.TEXTURE0+idx);
    GLOBAL_GL.bindTexture(GLOBAL_GL.TEXTURE_2D, this.texture);
    GLOBAL_GL.uniform1i(prgm.uniformLocationOf(name), idx);
  }

  del() {
    GLOBAL_GL.deleteTexture(this.texture);
  }
}

class DataTexture {
  constructor(data=_, width=256, height=width, idx=0, level=0) {
    this.data = data;
    this.width = width;
    this.height = height;
    this.idx = idx;

    this.level = level;

    this.update();
  }

  update(data=this.data, width=this.width, height=this.height, idx=this.idx, level=this.level) {
    try {
      this.del();
    } catch (e) {}

    this.data = data;
    this.width = width;
    this.height = height;
    this.idx = idx;
    this.level = level;

    this.texture = GLOBAL_GL.createTexture();
    GLOBAL_GL.activeTexture(GLOBAL_GL.TEXTURE0+idx);
    GLOBAL_GL.bindTexture(GLOBAL_GL.TEXTURE_2D, this.texture);
    GLOBAL_GL.texImage2D(GLOBAL_GL.TEXTURE_2D, level, GLOBAL_GL.RGBA, width, height, 0, GLOBAL_GL.RGBA, GLOBAL_GL.FLOAT, data||new Float32Array(width*height*4));

    GLOBAL_GL.texParameteri(GLOBAL_GL.TEXTURE_2D, GLOBAL_GL.TEXTURE_WRAP_S, GLOBAL_GL.CLAMP_TO_EDGE);
    GLOBAL_GL.texParameteri(GLOBAL_GL.TEXTURE_2D, GLOBAL_GL.TEXTURE_WRAP_T, GLOBAL_GL.CLAMP_TO_EDGE);
    GLOBAL_GL.texParameteri(GLOBAL_GL.TEXTURE_2D, GLOBAL_GL.TEXTURE_MIN_FILTER, GLOBAL_GL.NEAREST);
    GLOBAL_GL.texParameteri(GLOBAL_GL.TEXTURE_2D, GLOBAL_GL.TEXTURE_MAG_FILTER, GLOBAL_GL.NEAREST);
  }

  use(prgm, name, idx=this.idx) {
    GLOBAL_GL.activeTexture(GLOBAL_GL.TEXTURE0+idx);
    GLOBAL_GL.bindTexture(GLOBAL_GL.TEXTURE_2D, this.texture);
    GLOBAL_GL.uniform1i(prgm.uniformLocationOf(name), idx);
  }

  unbind() {
    GLOBAL_GL.bindTexture(GLOBAL_GL.TEXTURE_2D, null);
  }

  del() {
    GLOBAL_GL.deleteTexture(this.texture);
  }
}

class OrbitCamera {
  constructor(u=0, v=Math.PI/8, r=20, near=.01, far=50, zoomFactor=.9, rotSpeed=.002, rBoundary=[.001, 20]) {
    this.u = u;
    this.v = v;
    this.r = Math.min(Math.max(r, rBoundary[0]), rBoundary[1]);
    this.near = near;
    this.far = far;
    this.zoomFactor = zoomFactor;
    this.rotSpeed = rotSpeed;
    this.rBoundary = rBoundary;

    this.vBoundary = .5*Math.PI;
    this.mouseDown = 0;
    this.programs = [];
    this.text3D = [];
    this.calcPos();
    this.calcAxes();

    GLOBAL_CAMERA = this;

    this.rDestination = this.r;
    GLOBAL_CANVAS.onwheel = (e) => {
      e.preventDefault();
      if (e.deltaY < 0) {
        this.rDestination = Math.max(this.rDestination*this.zoomFactor, this.rBoundary[0]);
      } else {
        this.rDestination = Math.min(this.rDestination/this.zoomFactor, this.rBoundary[1]);
      }
      changeVal(this, "r", this.rDestination, 0, ()=>this.update(), _, _, false);
    };
    document.onmousemove = (e) => {
      if (this.mouseDown) {
        this.rotate(e.movementX, e.movementY);
      }
    };
    GLOBAL_CANVAS.onmousedown = () => {
      if (!document.fullscreenElement) document.body.requestFullscreen();
      this.mouseDown = 1;
    };
    document.onmouseup = () => {
      this.mouseDown = 0;
    };
    document.onkeydown = (e) => {
      if (e.key.includes("Arrow") && e.target.contains(GLOBAL_CANVAS)) {
        this.rotate(10*(e.key.includes("Left")-e.key.includes("Right")), 10*(e.key.includes("Up")-e.key.includes("Down")));
      }
    };
  }

  init() {}  // This does nothing

  rotate(du=0, dv=0) {
    this.u = this.u-du*this.rotSpeed;
    this.v = Math.max(Math.min(this.v+dv*this.rotSpeed, this.vBoundary), -this.vBoundary);
    this.calcPos();
    this.calcAxes();
    this.sendToShaders(["matrix", "camera"]);
    this.updateText3D();
  }

  update(u=this.u, v=this.v, r=this.r, sendAll=false) {
    this.u = u;
    this.v = v;
    this.r = this.rDestination = r;
    this.calcPos();
    this.calcAxes();
    if (sendAll) {
      this.sendToShaders();
    } else {
      this.sendToShaders(["matrix", "camera"]);
    }
    this.updateText3D();
  }

  calcPos() {
    let cosv = Math.cos(this.v);
    this.pos = [cosv*Math.cos(this.u), cosv*Math.sin(this.u), Math.sin(this.v)];
  }

  calcAxes() {
    let forward = this.pos.times(-1);
    let right = [forward[1], -forward[0], 0].normalized();
    let down = forward.cross(right);
    this.axes = [right, down, forward];
  }

  updateText3D() {
    this.text3D.forEach(text3D => {
      text3D.update();
    });
  }

  sendToShaders(select=["matrix", "camera", "near", "far", "coordMult"]) {
    this.programs.forEach(prgm => {
      if (select.includes("matrix")) {
        this.inverse = this.axes.inverse();
        GLOBAL_GL.uniformMatrix3fv(
          prgm.uniformLocationOf("matrix"),
          false,
          this.inverse.transpose().flat(1)
        );
      }
      if (select.includes("camera")) {
        GLOBAL_GL.uniform3fv(
          prgm.uniformLocationOf("camera"),
          this.pos.times(this.r)
        );
      }
      if (select.includes("near")) {
        GLOBAL_GL.uniform1f(
          prgm.uniformLocationOf("near"),
          this.near
        );
      }
      if (select.includes("far")) {
        GLOBAL_GL.uniform1f(
          prgm.uniformLocationOf("far"),
          this.far
        );
      }
      if (select.includes("coordMult")) {
        let a = GLOBAL_CANVAS.width/GLOBAL_CANVAS.height;
        this.coordMult = a<1?[1/a, -1]:[1, -a];
        GLOBAL_GL.uniform2fv(
          prgm.uniformLocationOf("coordMult"),
          this.coordMult
        );
      }
    });
  }
}

class Text3D {
  constructor(text, x=0, y=0, z=0, height=.2, color=[1, 1, 1]) {
    this.element = document.createElement("span");
    document.body.appendChild(this.element);
    this.element.style.position = "fixed";
    this.element.style.transform = "translate(-50%, -50%)";
    this.element.style.userSelect = "none";
    this.element.style.textAlign = "center";
    this.element.style.width = "10000px";
    this.element.style.pointerEvents = "none";

    GLOBAL_CAMERA.text3D.push(this);

    this.update(text, x, y, z, height, color);
  }

  update(text=this.text, x=this.x, y=this.y, z=this.z, height=this.height, color=this.color) {
    this.text = text;
    this.x = x;
    this.y = y;
    this.z = z;
    this.height = height
    this.color = color;
    this.element.innerHTML = text;
    let coordMult = .5*Math.max(GLOBAL_CANVAS.width, GLOBAL_CANVAS.height);
    this.pos = [[x, y, z].minus(GLOBAL_CAMERA.pos.times(GLOBAL_CAMERA.r))].matmul(GLOBAL_CAMERA.inverse).flat().times([coordMult, coordMult, 1]);
    this.pos = this.pos.over([this.pos[2], this.pos[2], 1]);
    if (this.pos[2] > GLOBAL_CAMERA.near && this.pos[2] < GLOBAL_CAMERA.far && !this.hidden) {
      this.element.style.display = "inline-block";
      this.element.style.left = `calc(50% + ${this.pos[0]}px)`;
      this.element.style.top = `calc(50% + ${this.pos[1]}px)`;
      let scale = 1/this.pos[2];
      this.element.style.fontSize = height*coordMult*scale+"px";
      this.element.style.color = `rgb(${String(color.times(255))})`;
    } else {
      this.element.style.display = "none";
    }
  }

  show() {
    this.hidden = false;
    this.element.style.display = "inline-block";
  }

  hide() {
    this.hidden = true;
    this.element.style.display = "none";
  }

  delete() {
    GLOBAL_CAMERA.text3D.filter(t => t !== this);
    this.element.remove();
  }
}

Array.prototype.cross = function(other) {
  return [this[1]*other[2]-this[2]*other[1], this[2]*other[0]-this[0]*other[2], this[0]*other[1]-this[1]*other[0]];
}

Array.prototype.inverse = function() {
  if (this.length === 3 && this[0].length === 3) {
    let minors = [[], [], []];
    let cofactors = [[], [], []];
    let adjugate = [[], [], []];
    let count = [0, 1, 2];
    for (let i of count) {
      for (let j of count) {
        let vals = [];
        for (let k of count.filter(n => n!==i)) {
          for (let l of count.filter(n => n!==j)) {
            vals.push(this[k][l]);
          }
        }
        minors[i][j] = vals[0]*vals[3]-vals[1]*vals[2];
        cofactors[i][j] = minors[i][j]*(i+j&1?-1:1);
        adjugate[j][i] = cofactors[i][j];
      }
    }
    return adjugate.map(arr => arr.map(n => n/(this[0][0]*minors[0][0]-this[0][1]*minors[0][1]+this[0][2]*minors[0][2])));
  }
  console.error("Inverse is optimized for and applies to only 3x3 matrices");
  return this;
}

Array.prototype.dot = function(other) {
  return this.map((n, i) => n*other[i]).reduce((a, b) => a+b);
};

Array.prototype.matmul = function(other) {
  let o = other.transpose();
  return this.map(a => other[0].slice()).map((a, i) => a.map((n, j) => this[i].dot(o[j])));
};

Array.prototype.plus = function(other) {
  return this.map((n, i) => n+(typeof(other) === "number"?other:other[i]));
};

Array.prototype.minus = function(other) {
  return this.map((n, i) => n-(typeof(other) === "number"?other:other[i]));
};

Array.prototype.times = function(other) {
  return this.map((n, i) => n*(typeof(other) === "number"?other:other[i]));
};

Array.prototype.over = function(other) {
  return this.map((n, i) => n/(typeof(other) === "number"?other:other[i]));
}

Array.prototype.magnitude = function() {
  return Math.sqrt(this.map(n => n**2).reduce((a, b) => a+b));
};

Array.prototype.sum = function() {
  return this.reduce((a, b) => a+b);
};

Array.prototype.normalized = function () {
  return this.over(this.magnitude());
};

Array.prototype.transpose = function() {
  let t = this.map(a => a.slice());
  this.forEach((a, i) => {
    a.forEach((n, j) => {
      t[j][i] = n;
    });
  });
  return t;
};

function changeVal(object, property, endValue, milliseconds=1000, onstep=()=>{}, onend=()=>{}, smooth=true) {
  let startValue = object[property];
  let valueChange = endValue-startValue;
  let dt = Math.min(10, .1*milliseconds);
  let steps = Math.floor(milliseconds/dt);
  let dValue = valueChange/steps;
  let t = performance.now();
  if (GLOBAL_INTERVALS[property]) {
    for (let key in GLOBAL_INTERVALS[property]) {
      if (GLOBAL_INTERVALS[property][key][0] === object) {
        GLOBAL_INTERVALS[property][key][2] = "clear";
      }
    }
  } else {
    GLOBAL_INTERVALS[property] = {};
  }
  let totalTime = 0;
  let key = "WRYIP24680~@$^*)+".split("").sort(() => Math.random()-.5).join("");
  let interval = setInterval(() => {
    if (totalTime === 0) {
      GLOBAL_INTERVALS[property][key] = [object, interval, "dontclear"];
    }
    if (GLOBAL_INTERVALS[property][key][2] === "clear") {
      clearInterval(interval);
      delete GLOBAL_INTERVALS[property][key];
    }
    try {
      let temp = performance.now();
      let elapsed = temp-t;
      totalTime += elapsed;
      t = temp;
      if (smooth) {
        object[property] = totalTime>=milliseconds?endValue:startValue+valueChange*.5*(Math.sin(Math.PI*(totalTime/milliseconds-.5))+1);
      } else {
        object[property] = [Math.max, Math.min][Number(dValue>0)](Number(object[property])+dValue*elapsed/dt, endValue);
      }
      onstep();
      if (object[property] === endValue) {
        clearInterval(interval);
        delete GLOBAL_INTERVALS[property][key];
        onend();
      }
    } catch (e) {
      clearInterval(interval);
      delete GLOBAL_INTERVALS[property][key];
      alert("changeVal error!\n"+e);
    }
  }, dt);
}

function drawArrow(startPos, endPos, color=[1, 1, 0], shadow=false) {
    if (!GLOBAL_ARROW) {
      GLOBAL_ARROW = {
        line: new Parametric1D(`return vec3(t, 0, 0);`, 0, 1, 1, .025, 16, [1, 0, 0]),
        tip: new Parametric2D(`return vec3(-u, .25*u*cos(v), .25*u*sin(v));`, 0, .2, .2, 0, 2*Math.PI, Math.PI/8, _, _, [1, 0, 0])
      };
      GLOBAL_ARROW.line.compute();
      GLOBAL_ARROW.tip.resolvePoles();
      GLOBAL_CAMERA.update(_, _, _, true);
    }
    if (!(JSON.stringify(color) === JSON.stringify(GLOBAL_ARROW.line.color) && JSON.stringify(color) === JSON.stringify(GLOBAL_ARROW.tip.color))) {
      GLOBAL_ARROW.line.update(_, _, _, _, _, color);
      GLOBAL_ARROW.tip.update(_, _, _, _, _, _, color);
    }
    let diff = endPos.minus(startPos);
    let factor = diff.magnitude();

    let zRot = Math.asin(diff[1]/factor);
    let yRot = -Math.atan2(diff[2], diff[0]);

    pushMatrix();
    scale(factor-.2);
    rotate(0, 0, zRot);
    rotate(0, yRot);
    translate(...startPos);
    if (shadow) GLOBAL_ARROW.line.drawShadow(...shadow);
    else GLOBAL_ARROW.line.draw();
    popMatrix();

    pushMatrix();
    rotate(0, 0, zRot);
    rotate(0, yRot);
    translate(...endPos);
    if (shadow) GLOBAL_ARROW.tip.drawShadow(...shadow);
    else GLOBAL_ARROW.tip.draw();
    popMatrix();
}

function clearScreen() {
  GLOBAL_GL.clear(GLOBAL_GL.COLOR_BUFFER_BIT | GLOBAL_GL.DEPTH_BUFFER_BIT);
}

performance.t = performance.now();
requestAnimationFrame(drawCaller);

function drawCaller() {
  try {
    draw;
    clearScreen();
    draw();
  } catch (e) { }

  let now = performance.now();
  performance.fps = 1000/(now-performance.t);
  performance.t = now;
  requestAnimationFrame(drawCaller);
}

/*
Everything below was added for the physics 171 project
*/

class ParProc {
  constructor(prgm) {
    this.prgm = prgm;
    this.framebuffer = GLOBAL_GL.createFramebuffer();
    this.computeVertices = new ComputeVertices;
    this.computeVertices.addAttributes(
      prgm,
      ["pos"],
      [2],
      [GLOBAL_GL.FLOAT],
      [0]
    );
  }

  run(params) {
    params.texIn.use(this.prgm, params.texInName);
    let outputWidth = params.outputWidth || params.texIn.width;
    let outputHeight = params.outputHeight || params.texIn.height;
    let texOut = new DataTexture(_, outputWidth, outputHeight, (params.texIn.idx+1)%16);

    GLOBAL_GL.bindFramebuffer(GLOBAL_GL.FRAMEBUFFER, this.framebuffer);
    GLOBAL_GL.framebufferTexture2D(GLOBAL_GL.FRAMEBUFFER, GLOBAL_GL.COLOR_ATTACHMENT0, GLOBAL_GL.TEXTURE_2D, texOut.texture, 0);
    GLOBAL_GL.viewport(0, 0, outputWidth, outputHeight);

    this.computeVertices.draw();

    params.texIn.del();
    params.texIn.texture = texOut.texture;

    GLOBAL_GL.viewport(0, 0, GLOBAL_CANVAS.width, GLOBAL_CANVAS.height);

    window.pixels = new Float32Array(4*outputWidth*outputHeight);
    // GLOBAL_GL.readPixels(0, 0, outputWidth, outputHeight, GLOBAL_GL.RGBA, GLOBAL_GL.FLOAT, pixels);
    GLOBAL_GL.bindFramebuffer(GLOBAL_GL.FRAMEBUFFER, null);
  }

  setUniform(name, value, methodString="uniform1f") {
    GLOBAL_GL[methodString](this.prgm.uniformLocationOf(name), value);
  }

  help() {
    console.log("The supplied program's vertes shader should look like this:");
    console.log(
      `precision mediump float;

      attribute vec2 pos;
      varying vec2 texPos;

      void main() {
        texPos = .5*(1.+pos);
        gl_Position = vec4(pos, 0, 1);
      }`
    );
  }
}

function dataTexResolveSize(data, idx=0) {
  let size = data.length/4;
  for (let width = Math.floor(Math.sqrt(size)); width >= 1; width--) {
    let height = size/width;
    if (height === Math.floor(height)) {
      return new DataTexture(new Float32Array(data), width, height, idx);
    }
  }
}

class P2D_Extra extends Vertices {
  constructor(r, rNormal, u0=0, u1=2*Math.PI, du=.1, v0=0, v1=2*Math.PI, dv=.1, color=[1, 0, 0]) {
    if (!PARAMETRIC_BUFFER) {
      PARAMETRIC_BUFFER = GLOBAL_GL.createBuffer();
      GLOBAL_GL.bindBuffer(GLOBAL_GL.ARRAY_BUFFER, PARAMETRIC_BUFFER);
      GLOBAL_GL.bufferData(GLOBAL_GL.ARRAY_BUFFER, new Float32Array(Array(1e7).keys()), GLOBAL_GL.STATIC_DRAW);
    }
    super(2*Math.round((u1-u0)/du+2)*Math.round((v1-v0)/dv));

    this.defaultDrawMode = GLOBAL_GL.TRIANGLE_STRIP;

    this.r = r;
    this.u0 = u0;
    this.u1 = u1;
    this.du = du;
    this.v0 = v0;
    this.v1 = v1;
    this.dv = dv;
    this.color = color;

    this.added = {};
    this.uCount = 2*Math.round((u1-u0)/du+2);
    this.vsSource = `precision mediump float;

      const float PI = 3.141592653589793;

      attribute float idx;

      uniform sampler2D dataTex;

      uniform vec2 coordMult;
      uniform mat3 matrix;
      uniform vec3 camera;
      uniform float near, far;

      uniform float u0, u1, du, v0, v1, dv;

      uniform float uCount;

      uniform float float0, float1, float2, float3, float4, float5, float6, float7, float8, float9, float10, float11, float12, float13, float14, float15;
      uniform vec4 vector0, vector1, vector2, vector3, vector4, vector5, vector6, vector7, vector8, vector9, vector10, vector11, vector12, vector13, vector14, vector15;
      uniform mat4 matrix0, matrix1, matrix2, matrix3, matrix4, matrix5, matrix6, matrix7, matrix8, matrix9, matrix10, matrix11, matrix12, matrix13, matrix14, matrix15;

      varying vec3 vNormal, vPos, vCam;

      uniform sampler2D tex0, tex1, tex2, tex3, tex4, tex5, tex6, tex7, tex8, tex9, tex10, tex11, tex12, tex13, tex14, tex15;
      uniform vec2 texDim0, texDim1, texDim2, texDim3, texDim4, texDim5, texDim6, texDim7, texDim8, texDim9, texDim10, texDim11, texDim12, texDim13, texDim14, texDim15;
      uniform vec3 rBounds0, rBounds1, rBounds2, rBounds3, rBounds4, rBounds5, rBounds6, rBounds7, rBounds8, rBounds9, rBounds10, rBounds11, rBounds12, rBounds13, rBounds14, rBounds15;

      uniform mat4 transformation;

      vec3 r0(float t) {
        float texIdx = floor((t-rBounds0.x)/rBounds0.z+.5);
        return texture2D(tex0, vec2(mod(texIdx, texDim0.x)+.5, floor(texIdx/texDim0.x)+.5)/texDim0).rgb;
      }

      vec3 r1(float t) {
        float texIdx = floor((t-rBounds1.x)/rBounds1.z+.5);
        return texture2D(tex1, vec2(mod(texIdx, texDim1.x)+.5, floor(texIdx/texDim1.x)+.5)/texDim1).rgb;
      }

      vec3 r2(float t) {
        float texIdx = floor((t-rBounds2.x)/rBounds2.z+.5);
        return texture2D(tex2, vec2(mod(texIdx, texDim2.x)+.5, floor(texIdx/texDim2.x)+.5)/texDim2).rgb;
        }

      vec3 r3(float t) {
        float texIdx = floor((t-rBounds3.x)/rBounds3.z+.5);
        return texture2D(tex3, vec2(mod(texIdx, texDim3.x)+.5, floor(texIdx/texDim3.x)+.5)/texDim3).rgb;
      }

      vec3 r4(float t) {
        float texIdx = floor((t-rBounds4.x)/rBounds4.z+.5);
        return texture2D(tex4, vec2(mod(texIdx, texDim4.x)+.5, floor(texIdx/texDim4.x)+.5)/texDim4).rgb;
      }

      vec3 r5(float t) {
        float texIdx = floor((t-rBounds5.x)/rBounds5.z+.5);
        return texture2D(tex5, vec2(mod(texIdx, texDim5.x)+.5, floor(texIdx/texDim5.x)+.5)/texDim5).rgb;
      }

      vec3 r6(float t) {
        float texIdx = floor((t-rBounds6.x)/rBounds6.z+.5);
        return texture2D(tex6, vec2(mod(texIdx, texDim6.x)+.5, floor(texIdx/texDim6.x)+.5)/texDim6).rgb;
      }

      vec3 r7(float t) {
        float texIdx = floor((t-rBounds7.x)/rBounds7.z+.5);
        return texture2D(tex7, vec2(mod(texIdx, texDim7.x)+.5, floor(texIdx/texDim7.x)+.5)/texDim7).rgb;
      }

      vec3 r8(float t) {
        float texIdx = floor((t-rBounds8.x)/rBounds8.z+.5);
        return texture2D(tex8, vec2(mod(texIdx, texDim8.x)+.5, floor(texIdx/texDim8.x)+.5)/texDim8).rgb;
      }

      vec3 r9(float t) {
        float texIdx = floor((t-rBounds9.x)/rBounds9.z+.5);
        return texture2D(tex9, vec2(mod(texIdx, texDim9.x)+.5, floor(texIdx/texDim9.x)+.5)/texDim9).rgb;
      }

      vec3 r10(float t) {
        float texIdx = floor((t-rBounds10.x)/rBounds10.z+.5);
        return texture2D(tex10, vec2(mod(texIdx, texDim10.x)+.5, floor(texIdx/texDim10.x)+.5)/texDim10).rgb;
      }

      vec3 r11(float t) {
        float texIdx = floor((t-rBounds11.x)/rBounds11.z+.5);
        return texture2D(tex11, vec2(mod(texIdx, texDim11.x)+.5, floor(texIdx/texDim11.x)+.5)/texDim11).rgb;
      }

      vec3 r12(float t) {
        float texIdx = floor((t-rBounds12.x)/rBounds12.z+.5);
        return texture2D(tex12, vec2(mod(texIdx, texDim12.x)+.5, floor(texIdx/texDim12.x)+.5)/texDim12).rgb;
      }

      vec3 r13(float t) {
        float texIdx = floor((t-rBounds13.x)/rBounds13.z+.5);
        return texture2D(tex13, vec2(mod(texIdx, texDim13.x)+.5, floor(texIdx/texDim13.x)+.5)/texDim13).rgb;
      }

      vec3 r14(float t) {
        float texIdx = floor((t-rBounds14.x)/rBounds14.z+.5);
        return texture2D(tex14, vec2(mod(texIdx, texDim14.x)+.5, floor(texIdx/texDim14.x)+.5)/texDim14).rgb;
      }

      vec3 r15(float t) {
        float texIdx = floor((t-rBounds15.x)/rBounds15.z+.5);
        return texture2D(tex15, vec2(mod(texIdx, texDim15.x)+.5, floor(texIdx/texDim15.x)+.5)/texDim15).rgb;
      }

      vec3 r(float u, float v) {
        ${r}
      }

      vec3 normalVector(float u, float v) {
        ${rNormal}
      }

      vec3 transformedR(float u, float v) {
        return (transformation*vec4(r(u, v), 1)).xyz;
      }

      vec4 proj3Dto2D(vec3 v) {
        vec3 new = (v-camera)*matrix;
        return vec4(new.xy*coordMult, 2.*(1.-new.z/near)/(1./far-1./near)-new.z, new.z);
      }

      void main() {
        float idxForRounding = idx+.5;
        float bandIdx = floor(idxForRounding/uCount);
        float bandPointIdx = min(max(floor(mod(idxForRounding, uCount))-1., 0.), uCount-3.);
        float u = u0+du*floor(.5*bandPointIdx+.1);
        float v = v0+dv*(bandIdx+mod(bandPointIdx, 2.));
        float uNext = u+du;
        float vNext = v+dv;

        vPos = transformedR(u, v);
        vNormal = normalize(normalVector(u, v));  // For a sphere
        vCam = camera;

        gl_Position = proj3Dto2D(vPos);
      }`;
    this.fsSource = `precision mediump float;

      uniform vec3 col;
      uniform bool shadow;

      varying vec3 vNormal, vPos, vCam;

      void main() {
        float ambient = .2;
        if (shadow) {
          gl_FragColor = vec4(ambient*col, 1);
        } else {
          vec3 lightDir1 = vec3(0, 0, -1);
          vec3 lightDir2 = vec3(-1, 0, 0);

          vec3 diff = vCam-vPos;

          vec3 normal = normalize(vNormal);
          if (dot(normal, vCam-vPos) < 0.) normal *= -1.;

          float diffuseStrength = .3;
          float diffuse = diffuseStrength*(max(0., dot(lightDir1, -normal))+max(0., dot(lightDir2, -normal)));

          float specularStrength = .5;
          vec3 viewDir = normalize(diff);
          vec3 reflectDir1 = reflect(lightDir1, normal);
          vec3 reflectDir2 = reflect(lightDir2, normal);
          float specular = specularStrength*(pow(max(0., dot(viewDir, reflectDir1)), 16.)+pow(max(0., dot(viewDir, reflectDir2)), 16.));

          vec3 result = (ambient+diffuse+specular)*col;
          gl_FragColor = vec4(result, 1);
        }
      }`;
    this.prgm = new Program(this.vsSource, this.fsSource);

    this.addAttributes(
      this.prgm,
      ["idx"],
      [1],
      [GLOBAL_GL.FLOAT],
      [0]
    );

    this.sendToShader();
  }

  compute() {}

  resolvePoles(ignore=[]) {
    this.shouldResolvePoles = true;

    let uSteps = Math.round((this.u1-this.u0)/this.du);
    let vSteps = Math.round((this.v1-this.v0)/this.dv);
    if (!ignore.includes("u0")) {this.u0 += .00001;}
    if (!ignore.includes("u1")) {this.u1 -= .00001;}
    if (!ignore.includes("v0")) {this.v0 += .00001;}
    if (!ignore.includes("v1")) {this.v1 -= .00001;}
    this.du = (this.u1-this.u0)/uSteps;
    this.dv = (this.v1-this.v0)/vSteps;

    this.uCount = 2*Math.round((this.u1-this.u0)/this.du+2);
    this.count = this.array = this.uCount*Math.round((this.v1-this.v0)/this.dv);  // Really specifying array size/count
    this.sendToShader();
  }

  update(u0=this.u0, u1=this.u1, du=this.du, v0=this.v0, v1=this.v1, dv=this.dv, color=this.color) {
    let ignore = [];
    if (this.shouldResolvePoles) {
      if (u0 === this.u0) {
        ignore.push("u0");
      } if (u1 === this.u1) {
        ignore.push("u1");
      } if (v0 === this.v0) {
        ignore.push("v0");
      } if (v1 === this.v1) {
        ignore.push("v1");
      }
    }
    this.u0 = u0;
    this.u1 = u1;
    this.du = du;
    this.v0 = v0;
    this.v1 = v1;
    this.dv = dv;
    this.color = color;
    if (this.shouldResolvePoles) {
      this.resolvePoles(ignore);
    } else {
      this.uCount = 2*Math.round((this.u1-this.u0)/this.du+2);
      this.count = this.array = this.uCount*Math.round((this.v1-this.v0)/this.dv);  // Really specifying array size/count
      this.sendToShader();
    }
  }

  drawShadow(direction="X", offset=0, color=[0, 0, 0]) {
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("shadow"), 1);
    GLOBAL_GL.uniform3fv(this.prgm.uniformLocationOf("col"), color);
    let scaling = ["x", "y", "z"].map(l => 1-(l === direction.toLowerCase()));
    let translation = scaling.map(n => offset*(1-n));
    pushMatrix();
    scale(...scaling);
    translate(...translation);
    this.draw();
    popMatrix();
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("shadow"), 0);
    GLOBAL_GL.uniform3fv(this.prgm.uniformLocationOf("col"), this.color);
  }

  setParametric1D(equationNumber, parametric1D, link={}) {
    let alreadyAdded = this.added[equationNumber];
    if (alreadyAdded) {  // unlink previous
      for (let key in alreadyAdded.computeParams.linked) {
        for (let i = 0; i < alreadyAdded.computeParams.linked[key].length; i++) {
          if (alreadyAdded.computeParams.linked[key][i].child === this) {
            alreadyAdded.computeParams.linked[key].splice(i, i+1);
          }
        }
      }
    }
    this.added[equationNumber] = parametric1D;
    for (let key in link) {
      if (!(key in parametric1D.computeParams.linked)) {
        parametric1D.computeParams.linked[key] = [];
      }
      parametric1D.computeParams.linked[key].push({child: this, variable: link[key]});
    }
  }

  setFloat(variableNumber, value) {
    if (variableNumber > 15) {
      alert("Highest variable number for setFloat is 15.");
      throw "Highest variable number for setFloat is 15.";
    }
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("float"+variableNumber), value);
  }

  setVector(variableNumber, value) {
    if (variableNumber > 15) {
      alert("Highest variable number for setVector is 15.");
      throw "Highest variable number for setVector is 15.";
    }
    if (value.length < 4) {
      alert("Must supply 4-component array with setVector!");
      throw "Must supply 4-component arrray with setVector!";
    }
    GLOBAL_GL.uniform4fv(this.prgm.uniformLocationOf("vector"+variableNumber), value);
  }

  setMatrix(variableNumber, value) {
    if (variableNumber > 15) {
      alert("Highest variable number for setMatrix is 15.");
      throw "Highest variable number for setMatrix is 15.";
    }
    if (value.length < 4) {
      alert("Must supply 4x4 array with setMatrix!");
      throw "Must supply 4x4 array with setMatrix!";
    }
    GLOBAL_GL.uniformMatrix4fv(this.prgm.uniformLocationOf("matrix"+variableNumber), value.transpose().flat(1));
  }

  sendToShader() {
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("u0"), this.u0);
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("u1"), this.u1);
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("du"), this.du);
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("v0"), this.v0);
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("v1"), this.v1);
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("dv"), this.dv);
    GLOBAL_GL.uniform3fv(this.prgm.uniformLocationOf("col"), this.color);
    GLOBAL_GL.uniform1f(this.prgm.uniformLocationOf("uCount"), this.uCount);
  }
}  // 0 A.B.
