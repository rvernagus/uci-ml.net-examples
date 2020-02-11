module Annealing
open Microsoft.ML.Data

[<CLIMutable>]
type AnnealingData =
    {
        [<LoadColumn(0)>]
        Family : string

        [<LoadColumn(1)>]
        ProductType : string

        [<LoadColumn(2)>]
        Steel : string

        [<LoadColumn(3)>]
        Carbon : float32

        [<LoadColumn(4)>]
        Hardness : float32

        [<LoadColumn(5)>]
        TemperRolling : string

        [<LoadColumn(6)>]
        Condition : string

        [<LoadColumn(7)>]
        Formability : string

        [<LoadColumn(8)>]
        Strength : float32

        [<LoadColumn(9)>]
        NonAgeing : string

        [<LoadColumn(10)>]
        SurfaceFinish : string

        [<LoadColumn(11)>]
        SurfaceQuality : string

        [<LoadColumn(12)>]
        Enamelability : string

        [<LoadColumn(13)>]
        Bc : string

        [<LoadColumn(14)>]
        Bf : string

        [<LoadColumn(15)>]
        Bt : string

        [<LoadColumn(16)>]
        BwMe : string

        [<LoadColumn(17)>]
        Bl : string

        [<LoadColumn(18)>]
        M : string

        [<LoadColumn(19)>]
        Chrom : string

        [<LoadColumn(20)>]
        Phos : string

        [<LoadColumn(21)>]
        Cbond : string

        [<LoadColumn(22)>]
        Marvi : string

        [<LoadColumn(23)>]
        Exptl : string

        [<LoadColumn(24)>]
        Ferro : string

        [<LoadColumn(25)>]
        Corr : string

        [<LoadColumn(26)>]
        BlueBrightVarnClean : string

        [<LoadColumn(27)>]
        Lustre : string

        [<LoadColumn(28)>]
        Jurofm : string

        [<LoadColumn(29)>]
        S : string

        [<LoadColumn(30)>]
        P : string

        [<LoadColumn(31)>]
        Shape : string

        [<LoadColumn(32)>]
        Thick : float32

        [<LoadColumn(33)>]
        Width : float32

        [<LoadColumn(34)>]
        Len : float32

        [<LoadColumn(35)>]
        Oil : string

        [<LoadColumn(36)>]
        Bore : string

        [<LoadColumn(37)>]
        Packing : string

        [<LoadColumn(38)>]
        [<ColumnName("Label")>]
        Classes : string
    }

[<CLIMutable>]
type AnnealingDataTransformed =
    {
        [<ColumnName("LabelValue")>]
        Classes : string

        [<VectorType(84)>]
        Features : single[]
    }

[<CLIMutable>]
type AnnealingPrediction =
    {
        LabelValue: string

        Score : single[]

        PredictedLabelValue : string
    }
