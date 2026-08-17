import { Mode } from "./types";
import { ranges } from "./utils";

export default class matrix {
    rows: number;
    cols: number;
    values: Array<Array<number>>

    constructor(ls: Array<Array<number>>) {
        this.values = ls
        this.rows = ls.length
        this.cols = ls[0].length
    }

    dotProduct(m2: matrix): matrix {
        if (this.rows !== m2.rows || this.cols !== m2.cols) throw Error("Dot Product incompatible")
        const mat: number[][] = []
        for (let i = 0; i < this.rows; i++) {
            const row: number[] = []
            for (let j = 0; j < this.cols; j++) {
                row.push(this.values[i][j] * m2.values[i][j])
            }
            mat.push(row)
        }
        return new matrix(mat)
    }

    window(x:number,y:number, scale:number = 3, mode: Mode = "Center"):matrix {

        const m: number[][] = Array.from({ length: scale }, () => Array(scale).fill(0.0));

        const anchors = ranges(scale, mode);

        const start_r = x - anchors.x
        const start_c = y - anchors.y

        for (let i=0; i<scale; i++) {
            const source_r:number = start_r + i
            for (let j=0; j<scale; j++) {
                const source_c:number = start_c + j
                if ((0 <= source_r && source_r < this.rows) && (0 <= source_c && source_c < this.cols)){
                    m[i][j] = this.values[source_r][source_c]
                } else m[i][j] = 0.0
            }
        }
        return new matrix(m);
    }

    conv2D(kernel: matrix, stride:number=1): matrix{
        const s = kernel.rows; 
        const kv = kernel.values; 
        const sv = this.values
        const out = []
        for (let i=0; i<this.rows - s + 1; i+=stride) {
            const row = [];
            for (let j=0; j<this.cols - s + 1; j+=stride) {
                let t = 0.0
                for (let a=0; a<s; a++) {
                    const ra = sv[i+a]; 
                    const ka = kv[a]
                    for (let b=0; b < s; b++) {
                        t += ra[j+b] * ka[b]
                    }
                }
                row.push(t)
            }
            out.push(row)
        }
        return new matrix(out)
    }

    reLu(): matrix {

        const m: number[][] = Array.from({ length: this.rows }, () => Array(this.cols).fill(0.0));

        for (let i=0; i<this.rows; i++) {
            for (let j=0; j< this.cols; j++) {
                m[i][j] = Math.max(this.values[i][j], 0.0)
            }
        }
        return new matrix(m)
    }

    sum(): number {
        let s = 0;
        for (let i =0; i<this.rows; i++) {
            for(let j=0; j<this.cols; j++) {
                s+=this.values[i][j]
            }
        }
        return s
    }

    _max(): number {
        let curr = this.values[0][0];
        for (let i =0; i<this.rows; i++) {
            for(let j=0; j<this.cols; j++) {
                curr = Math.max(curr, this.values[i][j])
            }
        }
        return curr;
    }

    maxPool(k:number=2):matrix {
        const out_r = Math.floor(this.rows / 2);
        const out_c = Math.floor(this.cols / 2);
        const out: number[][] = Array.from({ length: out_r }, () => Array(out_c).fill(0.0));
        for (let i =0; i<out_r; i++) {
            for(let j=0; j<out_r; j++) {
                out[i][j] = this.window(i * k, j * k, k, "Top")._max()
            }
        }
        return new matrix(out);
    }

    add(m2:matrix):matrix {
        const out: number[][] = Array.from({ length: this.rows }, () => Array(this.cols).fill(0.0));
        for (let i =0; i<this.rows; i++) {
            for(let j=0; j<this.cols; j++) {
                out[i][j] = this.values[i][j] + m2.values[i][j]
            }
        }
        return new matrix(out);
    }

    flatten():Array<number> {
        const r = []
        for (let i =0; i<this.rows; i++) {
            for(let j=0; j<this.cols; j++) {
                r.push(this.values[i][j])
            }
        }
        return r;
    }

    mult(b: matrix) {
    if (this.cols !== b.rows) {
        throw new Error("Matrices can not multiplied C1 != R2");
    }
    
    const zeroGrid = Array.from({ length: this.rows }, () => 
        new Array(b.cols).fill(0.0)
    );
    
    const ret = new matrix(zeroGrid);
    
    for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < b.cols; j++) {
            let s = 0.0;
            for (let k = 0; k < this.cols; k++) {
                s += this.values[i][k] * b.values[k][j];
            }
            ret.values[i][j] = s; 
        }
    }
            
    return ret;
}

}