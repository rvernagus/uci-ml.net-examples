using Microsoft.ML.Data;

namespace Adult
{
    public class AdultData
    {
        [LoadColumn(0)]
        public float Age { get; set; }

        [LoadColumn(1)]
        public string WorkClass { get; set; }

        [LoadColumn(2)]
        public float Fnlwgt { get; set; }

        [LoadColumn(3)]
        public string Education { get; set; }

        [LoadColumn(4)]
        public float EducationNum { get; set; }

        [LoadColumn(5)]
        public float MaritalStatus { get; set; }

        [LoadColumn(6)]
        public float Occupation { get; set; }

        [LoadColumn(7)]
        public float Relationship { get; set; }

        [LoadColumn(8)]
        public string Race { get; set; }

        [LoadColumn(9)]
        public string Sex { get; set; }

        [LoadColumn(10)]
        public float CapitalGain { get; set; }

        [LoadColumn(11)]
        public float CapitalLoss { get; set; }

        [LoadColumn(12)]
        public float HoursPerWeek { get; set; }

        [LoadColumn(13)]
        public string NativeCountry { get; set; }

        [LoadColumn(14)]
        [ColumnName("Label")]
        public string Target { get; set; }

        public override string ToString() =>
            $"{{ Age: {Age}\n  WorkClass: {WorkClass}\n  Fnlwgt: {Fnlwgt}\n  Education: {Education}\n  EducationNum: {EducationNum}\n  MaritalStatus: {MaritalStatus}\n  Occupation: {Occupation}\n  Relationship: {Relationship}\n  Race: {Race}\n  Sex: {Sex}\n  CapitalGain: {CapitalGain}\n  CapitalLoss: {CapitalLoss}\n  HoursPerWeek: {HoursPerWeek}\n  NativeCountry: {NativeCountry}\n  Target: {Target} }}";
    }

    public class AdultDataTransformed
    {
        [ColumnName("Label")]
        public bool Target { get; set; }

        [VectorType(83)]
        public float[] Features { get; set; }

        [VectorType(83)]
        public float[] FeaturesNorm { get; set; }

        public override string ToString() =>
            $"{{ Target: {Target}\n  Features: {string.Join(",", Features)}\n  FeaturesNorm: {string.Join(",", FeaturesNorm)} }}";
    }
}
