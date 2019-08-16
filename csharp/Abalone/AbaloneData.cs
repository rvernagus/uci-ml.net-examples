using Microsoft.ML.Data;
using System;

namespace Abalone
{
    public class AbaloneData
    {
        [LoadColumn(0)]
        public string Sex { get; set; }

        [LoadColumn(1)]
        public float Length { get; set; }

        [LoadColumn(2)]
        public float Diameter { get; set; }

        [LoadColumn(3)]
        public float Height { get; set; }

        [LoadColumn(4)]
        public float WholeWeight { get; set; }

        [LoadColumn(5)]
        public float ShuckedWeight { get; set; }

        [LoadColumn(6)]
        public float VisceraWeight { get; set; }

        [LoadColumn(7)]
        public float ShellWeight { get; set; }

        [LoadColumn(8)]
        [ColumnName("Label")]
        public Single Rings { get; set; }
    }
}
