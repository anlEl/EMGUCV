using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.CV.Util;
using Emgu.CV.XFeatures2D;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace EmguCVTest
{
    class Program
    {
        static void Main(string[] args)
        {
            ImageViewer viewer = new ImageViewer(); //create an image viewer
            Capture capture = new Capture(); //create a camera captue
            FastDetector detector = new FastDetector(10);
            Application.Idle += new EventHandler(delegate (object sender, EventArgs e)
            {  //run this until application closed (close button click on image viewer)
                Mat frame   = capture.QueryFrame();
                VectorOfKeyPoint keypoints = new VectorOfKeyPoint( detector.Detect(frame));
                Features2DToolbox.DrawKeypoints(frame, keypoints, frame, new Bgr(System.Drawing.Color.Red), Features2DToolbox.KeypointDrawType.Default);
                viewer.Image = frame; //draw the image obtained from camera
            });
            viewer.ShowDialog(); //show the image viewer
        }
        public static Image<Bgr, Byte> Draw(Image<Gray, Byte> modelImage, Image<Gray, byte> observedImage)
        {
            Mat homography = null;

            FastDetector fastCPU = new FastDetector(15, true);
            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;
            VectorOfVectorOfDMatch indices = new VectorOfVectorOfDMatch();

            BriefDescriptorExtractor descriptor = new BriefDescriptorExtractor();

            Mat mask;
            int k = 2;
            double uniquenessThreshold = 0.8;

            //extract features from the object image
            modelKeyPoints = new VectorOfKeyPoint(fastCPU.Detect(modelImage));
            UMat modelDescriptors = new UMat();
            descriptor.Compute(modelImage, modelKeyPoints, modelDescriptors);

            // extract features from the observed image
            observedKeyPoints = new VectorOfKeyPoint(fastCPU.Detect(observedImage));
            UMat observedDescriptors = new UMat();
            descriptor.Compute(observedImage, observedKeyPoints, observedDescriptors);
            BFMatcher matcher = new BFMatcher(DistanceType.L2);
            matcher.Add(modelDescriptors);

            using (VectorOfVectorOfDMatch dist = new VectorOfVectorOfDMatch(observedDescriptors.Rows))
            {
                matcher.KnnMatch(observedDescriptors, indices, k, null);
                //matcher.KnnMatch(observedDescriptors, indices, dist, k, null);
                mask = new Mat(modelImage.Size,DepthType.Cv8U,modelImage.NumberOfChannels);
                mask.SetTo(new MCvScalar(255));
                Features2DToolbox.VoteForUniqueness(dist, uniquenessThreshold, mask);
            }

            int nonZeroCount = CvInvoke.CountNonZero(mask);
            if (nonZeroCount >= 4)
            {
                nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, indices, mask, 1.5, 20);
                if (nonZeroCount >= 4)
                    homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints, observedKeyPoints, indices, mask, 2);
            }

            //Draw the matched keypoints
            Image<Bgr, Byte> result = new Image<Bgr, byte>(modelImage.Size);
                Features2DToolbox.DrawMatches(modelImage, modelKeyPoints, observedImage, observedKeyPoints,
               indices, result, new MCvScalar(255, 255, 255), new MCvScalar(255, 255, 255), mask, Features2DToolbox.KeypointDrawType.Default);

            #region draw the projected region on the image
            if (homography != null)
            {  //draw a rectangle along the projected model
                Rectangle rect = modelImage.ROI;
                PointF[] pts = new PointF[] {
                new PointF(rect.Left, rect.Bottom),
                new PointF(rect.Right, rect.Bottom),
                new PointF(rect.Right, rect.Top),
                new PointF(rect.Left, rect.Top)};

                //CvInvoke.ProjectPoints(pts, homography);
                //homography.ProjectPoints(pts);

                result.DrawPolyline(Array.ConvertAll<PointF, Point>(pts, Point.Round), true, new Bgr(Color.Red), 5);
            }
            #endregion

            return result;
        }

    }
}
