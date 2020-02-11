namespace Arrhythmia
{
    public class ArrhythmiaDataTransformed
    {
        public int LabelValue { get; set; }

        public float[] Features { get; set; }

        public override string ToString() =>
            $"{{ LabelValue: {LabelValue}\n  Features: {string.Join(",", Features)} }}";
    }
}
