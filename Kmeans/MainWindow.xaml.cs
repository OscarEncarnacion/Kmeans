using Kmeans.Classes;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using System.Windows;

namespace Kmeans
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private List<Dato> listaDatos;
        private Dictionary<int, int> coloresPorCluster = new Dictionary<int, int>();

        public MainWindow()
        {
            InitializeComponent();
            listaDatos =
            [
                new Dato(2f, 4f),
                new Dato(5f, 7f),
                new Dato(3f, 5f),
                new Dato(1f, 3f),

                new Dato(26f, 35f),
                new Dato(24f, 31f),
                new Dato(22f, 32f),
                new Dato(27f, 34f),

                new Dato(15f, 19f),
                new Dato(13f, 16f),
                new Dato(14f, 12f),
                new Dato(12f, 14f),

                new Dato(43f, 45f),
                new Dato(39f, 42f),
                new Dato(46f, 38f),
                new Dato(44f, 40f)
            ];
            DatosDataGrid.ItemsSource = listaDatos;
        }

        private void ClasificarButtonClick(object sender, RoutedEventArgs e)
        {
            if (int.TryParse(KTextBox.Text, out int k) && int.TryParse(IteracionesTextBox.Text, out int iteraciones))
            {
                if (k > 0 && iteraciones > 50)
                {
                    CalcularClusters(k, iteraciones);
                }
                else
                {
                    MessageBox.Show("K debe ser mayor a 0 e Iteraciones debe ser mayore a 50");
                }
            }
            else
            {
                MessageBox.Show("K e Iteraciones deben ser un números enteros");
            }
        }

        private void CalcularClusters(int k, int iteraciones)
        {
            var context = new MLContext();
            var data = context.Data.LoadFromEnumerable(listaDatos);
            var options = new KMeansTrainer.Options
            {
                NumberOfClusters = k,
                MaximumNumberOfIterations = iteraciones
            };
            var pipeline = context.Transforms.Concatenate("Features", "X", "Y").Append(context.Clustering.Trainers.KMeans(options));
            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);
            var clusterIdColumn = predictions.GetColumn<uint>("PredictedLabel").ToArray();

            var random = new Random(314);
            for (int i = 1; i <= k; i++)
            {
                coloresPorCluster.Add(i, random.Next(100, 1000));
            }

            var grafica = new PlotModel { Title = "Grafica" };
            var scatterSeries = new ScatterSeries { MarkerType = MarkerType.Circle };

            string temp = "";
            for (int i = 0; i < listaDatos.Count; i++)
            {
                var scatterPoint = new ScatterPoint(listaDatos[i].X, listaDatos[i].Y, 5, coloresPorCluster[int.Parse(clusterIdColumn[i].ToString())]);
                scatterSeries.Points.Add(scatterPoint);
                temp += $"X: {listaDatos[i].X} Y: {listaDatos[i].Y} Cluster: {clusterIdColumn[i]}\n";
            }
            grafica.Series.Add(scatterSeries);
            grafica.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Eje X" });
            grafica.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Eje Y" });
            grafica.Axes.Add(new LinearColorAxis { Position = AxisPosition.Right, Palette = OxyPalettes.Jet(200) });
            Plot.Model = grafica;
            ClustersTextBox.Text = temp;
            coloresPorCluster.Clear();
        }
    }
}