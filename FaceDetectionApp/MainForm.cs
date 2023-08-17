using OpenCvSharp;
using OpenCvSharp.Extensions;
using System.Timers;
using System.Windows.Forms;

namespace FaceDetectionApp
{
    public partial class MainForm : Form
    {
        private VideoCapture capture;
        private CascadeClassifier faceCascade;
        private CascadeClassifier smileCascade;
        private System.Timers.Timer processingTimer;
        public MainForm()
        {
            InitializeComponent();
            capture = new VideoCapture(0);
            faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");
            smileCascade = new CascadeClassifier("haarcascade_smile.xml");

            processingTimer = new System.Timers.Timer();
            processingTimer.Interval = 100; 
            processingTimer.Elapsed += ProcessFrame; 
            processingTimer.Start(); 
        }
        private void ProcessFrame(object sender, ElapsedEventArgs e)
        {
            Mat frame = new Mat();
            capture.Read(frame);

            if (!frame.Empty())
            {
                Mat grayFrame = new Mat();
                Cv2.CvtColor(frame, grayFrame, ColorConversionCodes.BGR2GRAY);

                Rect[] faces = faceCascade.DetectMultiScale(grayFrame);

                foreach (Rect face in faces)
                {
                    bool hasOverlap = false;

                    Mat faceROI = grayFrame.SubMat(face);
                    Rect[] smiles = smileCascade.DetectMultiScale(faceROI, 1.8, 20, 0 | HaarDetectionTypes.ScaleImage, new Size(20, 20));

                    foreach (Rect smile in smiles)
                    {
                        Point smileLocation = new Point(face.X + smile.X, face.Y + smile.Y);
                        Size smileSize = new Size(smile.Width, smile.Height);

                        if (smile.IntersectsWith(face))
                        {
                            hasOverlap = true;
                            break;
                        }

                        Cv2.Rectangle(frame, new Rect(smileLocation, smileSize), Scalar.Green, 2);
                    }

                    if (!hasOverlap)
                    {
                        Cv2.Rectangle(frame, face, Scalar.Red, 2);
                    }
                }

                pictureBox1.Image = BitmapConverter.ToBitmap(frame);
            }
        }
    }
}
