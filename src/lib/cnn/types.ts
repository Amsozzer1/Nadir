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