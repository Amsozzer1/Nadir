import matrix from "./matrix"

export type Mode = "Top" | "Center" | "Bottom"

export type r = {
    x: number,
    y: number,
}
export type Layer = {
    filters: Array<Array<matrix>>,
    stride: number
}

// Every stage of one conv layer, captured on the way through so the UI can
// replay the computation without re-running it. `pre` still has negatives in
// it -- that is the only reason the ReLU step is showable at all.
export type Activation = {
    layer: number,
    pre: Array<matrix>,
    maps: Array<matrix>,
    pooled: Array<matrix>,
}

export type Trace = Array<Activation>

// Shape of the JSON written by the trainer's save_checkpoint.
export type Checkpoint = {
    epoch: number,
    shapes: Array<[number, number]>,
    weights: Array<Array<Array<number>>>,
    biases: Array<Array<number>>,
    kernels: Array<Array<Array<Array<Array<number>>>>>,
}

export type Model = {
    epoch: number,
    layers: Array<Layer>,
    W1: matrix,
    b1: Array<number>,
    W2: matrix,
    b2: Array<number>,
}