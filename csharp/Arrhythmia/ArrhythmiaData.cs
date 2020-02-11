namespace Arrhythmia
{
    public class ArrhythmiaDataTransformed
    {
        public int LabelValue { get; set; }

        public float[] FeaturesPCA { get; set; }

        public override string ToString() =>
            $"{{ LabelValue: {LabelValue}\n  FeaturesPCA: {string.Join(",", FeaturesPCA)} }}";
    }
}
