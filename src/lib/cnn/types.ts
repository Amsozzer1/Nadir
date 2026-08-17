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