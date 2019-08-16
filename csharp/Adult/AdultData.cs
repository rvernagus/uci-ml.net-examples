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
    }
}
