using Microsoft.ML.Data;

namespace Annealing
{
    public class AnnealData
    {
        [LoadColumn(0)]
        public string Family { get; set; }

        [LoadColumn(1)]
        public string ProductType { get; set; }

        [LoadColumn(2)]
        public string Steel { get; set; }

        [LoadColumn(3)]
        public float Carbon { get; set; }

        [LoadColumn(4)]
        public float Hardness { get; set; }

        [LoadColumn(5)]
        public string TemperRolling { get; set; }

        [LoadColumn(6)]
        public string Condition { get; set; }

        [LoadColumn(7)]
        public string Formability { get; set; }

        [LoadColumn(8)]
        public float Strength { get; set; }

        [LoadColumn(9)]
        public string NonAgeing { get; set; }

        [LoadColumn(10)]
        public string SurfaceFinish { get; set; }

        [LoadColumn(11)]
        public string SurfaceQuality { get; set; }

        [LoadColumn(12)]
        public string Enamelability { get; set; }

        [LoadColumn(13)]
        public string Bc { get; set; }

        [LoadColumn(14)]
        public string Bf { get; set; }

        [LoadColumn(15)]
        public string Bt { get; set; }

        [LoadColumn(16)]
        public string BwMe { get; set; }

        [LoadColumn(17)]
        public string Bl { get; set; }

        [LoadColumn(18)]
        public string M { get; set; }

        [LoadColumn(19)]
        public string Chrom { get; set; }

        [LoadColumn(20)]
        public string Phos { get; set; }

        [LoadColumn(21)]
        public string Cbond { get; set; }

        [LoadColumn(22)]
        public string Marvi { get; set; }

        [LoadColumn(23)]
        public string Exptl { get; set; }

        [LoadColumn(24)]
        public string Ferro { get; set; }

        [LoadColumn(25)]
        public string Corr { get; set; }

        [LoadColumn(26)]
        public string BlueBrightVarnClean { get; set; }

        [LoadColumn(27)]
        public string Lustre { get; set; }

        [LoadColumn(28)]
        public string Jurofm { get; set; }

        [LoadColumn(29)]
        public string S { get; set; }

        [LoadColumn(30)]
        public string P { get; set; }

        [LoadColumn(31)]
        public string Shape { get; set; }

        [LoadColumn(32)]
        public float Thick { get; set; }

        [LoadColumn(33)]
        public float Width { get; set; }

        [LoadColumn(34)]
        public float Len { get; set; }

        [LoadColumn(35)]
        public string Oil { get; set; }

        [LoadColumn(36)]
        public string Bore { get; set; }

        [LoadColumn(37)]
        public string Packing { get; set; }

        [LoadColumn(38)]
        [ColumnName("Label")]
        public string Classes { get; set; }
    }

    public class AnnealDataTransformed
    {
        [ColumnName("LabelValue")]
        public string Classes { get; set; }

        [VectorType(84)]
        public float[] Features { get; set; }

        public override string ToString() =>
            $"{{ Classes: {Classes}\n  Features: {string.Join(",", Features)} }}";
    }
}
