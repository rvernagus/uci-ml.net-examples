module Autos
open Microsoft.ML.Data

[<CLIMutable>]
type AutosData =
    {
        [<LoadColumn(0)>]
        Symboling : string
        [<LoadColumn(1)>] 
        NormLosses : float32
        [<LoadColumn(2)>]
        Make : string
        [<LoadColumn(3)>]
        FuelType : string
        [<LoadColumn(4)>]
        Aspiration : string
        [<LoadColumn(5)>]
        NumDoors : string
        [<LoadColumn(6)>]
        BodyStyle : string
        [<LoadColumn(7)>]
        DriveWheels : string
        [<LoadColumn(8)>]
        EngineLoc : string
        [<LoadColumn(9)>]
        WheelBase : float32
        [<LoadColumn(10)>]
        Length : float32
        [<LoadColumn(11)>]
        Width : float32
        [<LoadColumn(12)>]
        Height : float32
        [<LoadColumn(13)>]
        CurbWeight : float32
        [<LoadColumn(14)>]
        EngineType : string
        [<LoadColumn(15)>]
        NumCylinders : string
        [<LoadColumn(16)>]
        EngineSize : float32
        [<LoadColumn(17)>]
        FuelSystem : string
        [<LoadColumn(18)>]
        Bore : float32
        [<LoadColumn(19)>]
        Stroke : float32
        [<LoadColumn(20)>]
        CompressionRatio : float32
        [<LoadColumn(21)>]
        Horsepower : float32
        [<LoadColumn(22)>]
        PeakRpm : float32
        [<LoadColumn(23)>]
        CityMpg : float32
        [<LoadColumn(24)>]
        HighwayMpg : float32
        [<LoadColumn(25)>]
        [<ColumnName("Label")>]
        Price : float32
    }