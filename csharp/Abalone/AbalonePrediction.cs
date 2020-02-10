using Microsoft.ML.Data;
using System;

namespace Abalone
{
    public class AbalonePrediction
    {
        [ColumnName("Label")]
        public Single ActualRings { get; set; }

        [ColumnName("Score")]
        public Single PredictedRings { get; set; }

        public override string ToString() =>
            $"ActualRings: {ActualRings}, PredictedRings: {PredictedRings}";
    }
}
