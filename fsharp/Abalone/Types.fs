module Abalone
open Microsoft.ML.Data

[<CLIMutable>]
type AbaloneData =
    {
        [<LoadColumn(0)>]
        Sex : string
        [<LoadColumn(1)>]
        Length : float32
        [<LoadColumn(2)>]
        Diameter : float32
        [<LoadColumn(3)>]
        Height : float32
        [<LoadColumn(4)>]
        WholeWeight : float32
        [<LoadColumn(5)>]
        ShuckedWeight : float32
        [<LoadColumn(6)>]
        VisceraWeight : float32
        [<LoadColumn(7)>]
        ShellWeight : float32
        [<LoadColumn(8)>]
        [<ColumnName("Label")>]
        Rings : single
    }


[<CLIMutable>]
type AbaloneDataTransformed =
    {
        Sex : single[]
        Length : float32
        Diameter : float32
        Height : float32
        WholeWeight : float32
        ShuckedWeight : float32
        VisceraWeight : float32
        ShellWeight : float32
        Label : single
        Features : single[]
        FeaturesNorm : single[]
    }


[<CLIMutable>]
type AbalonePrediction =
    {
        Label : single
        Score : single
    }
