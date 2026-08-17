import matrix from "./matrix"
import { Layer, Mode, r } from "./types"

export function ranges(scale: number, mode: Mode): r {
    const middle = Math.floor(scale / 2)
    const last = scale - 1
    const ret: r = {
        x: middle,
        y: middle
    }
    if (mode === "Top") {
        ret.x = 0
        ret.y = 0
    }
    if (mode === "Bottom") {
        ret.x = last
        ret.y = last
    }
    return ret
}

export function feature_extraction(
    layers: Array<Layer>, 
    curr: Array<matrix>, 
) {

    for (let i=0; i<layers.length; i++) {
        const out = []
        const filters = layers[i].filters
        const stride = layers[i].stride
        for (let j=0; j<filters.length; j++) {
            const stack = filters[j]
            let acc:matrix|null = null;
            for (let k=0; k<Math.min(curr.length, stack.length); k++) {
                const ch_map = curr[k]
                const kern = stack[k]
                const m = ch_map.conv2D(kern, stride)
                acc = acc ? acc.add(m) : m
            }
            out.push(acc)
        }
        curr = []
        for (let l=0; l<out.length; l++) {
            const z = out[l]
            if (!z) throw Error("Z cant be null")
            curr.push(z?.reLu().maxPool(2))
        }
    }
    return curr;
}

function flatten_3d(input: Array<matrix>): Array<number> {
    let r:Array<number> = []
    for (let m=0; m<input.length; m++) {
        const mat: matrix = input[m]
        const f:Array<number>  = mat.flatten()
        r = [...r,...f]
    }
    return r
}
function dense(x: Array<number>, W: matrix, b: Array<number>) {
    const X = new matrix(x.map(row => [row]));
    const B = new matrix(b.map(row => [row]));
    return W.mult(X).add(B).flatten();
}
function relu_vec(x: Array<number>) {
    return x.map(v => Math.max(0.0, v));
}
function softmax(z: Array<number>) {
    const hi = Math.max(...z);
    const e = z.map(v => Math.exp(v - hi));
    const t = e.reduce((sum, v) => sum + v, 0);
    return e.map(v => v / t);
}

export function classification(
    features: Array<matrix>,
    W1: matrix,
    b1: Array<number>,
    W2: matrix,
    b2:Array<number>): Array<number> {
        const flattened_features = flatten_3d(features);
        const l1 = dense(flattened_features, W1, b1);
        const h = relu_vec(l1)
        const l2 = dense(h, W2, b2)
        const probs = softmax(l2)
        return probs;
    }