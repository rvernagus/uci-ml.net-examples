module Adult
open Microsoft.ML.Data


[<CLIMutable>]
type AdultData =
    {
        [<LoadColumn(0)>]
        Age : float32
        [<LoadColumn(1)>]
        WorkClass : string
        [<LoadColumn(2)>]
        Fnlwgt : float32
        [<LoadColumn(3)>]
        Education : string
        [<LoadColumn(4)>]
        EducationNum : float32
        [<LoadColumn(5)>]
        MaritalStatus : float32
        [<LoadColumn(6)>]
        Occupation : float32
        [<LoadColumn(7)>]
        Relationship : float32
        [<LoadColumn(8)>]
        Race : string
        [<LoadColumn(9)>]
        Sex : string
        [<LoadColumn(10)>]
        CapitalGain : float32
        [<LoadColumn(11)>]
        CapitalLoss : float32
        [<LoadColumn(12)>]
        HoursPerWeek : float32
        [<LoadColumn(13)>]
        NativeCountry : string
        [<LoadColumn(14)>]
        [<ColumnName("Label")>]
        Target : string
    }
