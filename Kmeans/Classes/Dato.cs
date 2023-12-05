namespace Kmeans.Classes
{
    internal class Dato
    {
        public float X { get; set; }
        public float Y { get; set; }

        public Dato()
        {
            X = 0;
            Y = 0;
        }

        public Dato(float x, float y)
        {
            X = x;
            Y = y;
        }
    }
}
