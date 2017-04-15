using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.CV.Util;
using Emgu.CV.XFeatures2D;
using System;
using System.Drawing;
using System.Windows.Forms;

namespace EmguCVTest
{
    class FastDrawer : IDisposable
    {
        private Mat                         homography;
        private FastDetector                fastCPU;
        private VectorOfKeyPoint            modelKeyPoints;
        private VectorOfKeyPoint            observedKeyPoints;
        private VectorOfVectorOfDMatch      indices;
        private BriefDescriptorExtractor    descriptor;
        private Mat                         mask;
        private int                         k;
        private double                      uniquenessThreshold;
        private Mat                         modelDescriptors;
        private Mat                         observedDescriptors;
        private BFMatcher                   matcher;
        private Mat                         result ;
        public FastDrawer()
        {
            homography          = null;
            fastCPU             = new FastDetector(15);
            indices             = new VectorOfVectorOfDMatch();
            descriptor          = new BriefDescriptorExtractor();
            k                   = 2;
            uniquenessThreshold = 0.8;
            modelDescriptors    = new Mat();
            observedDescriptors = new Mat();
            matcher             = new BFMatcher(DistanceType.L2);
            result              = new Mat();
        }
        public Image<Bgr, Byte> Draw(Image<Bgr, Byte> modelImage, Image<Bgr, Byte> observedImage)
        {

            //extract features from the object image
            modelKeyPoints = new VectorOfKeyPoint(fastCPU.Detect(modelImage));
            descriptor.Compute(modelImage, modelKeyPoints, modelDescriptors);

            // extract features from the observed image
            observedKeyPoints = new VectorOfKeyPoint(fastCPU.Detect(observedImage));
            descriptor.Compute(observedImage, observedKeyPoints, observedDescriptors);

            if (modelKeyPoints.Size == 0 || observedKeyPoints.Size == 0)
                return observedImage;
            matcher.Add(modelDescriptors);


            matcher.KnnMatch(observedDescriptors, indices, k, null);
            mask = new Mat(observedDescriptors.Size, observedDescriptors.Depth, observedDescriptors.NumberOfChannels);
            mask.SetTo(new MCvScalar(255));
            Features2DToolbox.VoteForUniqueness(indices, uniquenessThreshold, mask);


            int nonZeroCount = CvInvoke.CountNonZero(mask);
            if (nonZeroCount >= 4)
            {
                nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, indices, mask, 1.5, 20);
                if (nonZeroCount >= 4)
                    homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints, observedKeyPoints, indices, mask, 2);
            }

            //Draw the matched keypoints
            Features2DToolbox.DrawMatches(modelImage, modelKeyPoints, observedImage, observedKeyPoints,
            indices, result, new MCvScalar(255, 200, 150), new MCvScalar(0, 0, 255), null, Features2DToolbox.KeypointDrawType.DrawRichKeypoints);
            Image<Bgr, byte> res = result.ToImage<Bgr, byte>();
            #region draw the projected region on the image
            if (homography != null)
            {  //draw a rectangle along the projected model
                Rectangle rect = res.ROI;
                PointF[] pts = new PointF[] {
                new PointF(rect.Left, rect.Bottom),
                new PointF(rect.Right, rect.Bottom),
                new PointF(rect.Right, rect.Top),
                new PointF(rect.Left, rect.Top)};

                //CvInvoke.ProjectPoints(pts , homography);
                //homography.ProjectPoints(pts);

                res.DrawPolyline(Array.ConvertAll<PointF, Point>(pts, Point.Round), true, new Bgr(Color.Red), 5);
            }
            #endregion
            modelImage.Dispose();
            return res;
        }

        public void Dispose()
        {
            k                   = 2;
            uniquenessThreshold = 0.8;
            homography          = null;
            fastCPU             = null;
            indices             = null;
            descriptor          = null;
            modelDescriptors    = null;
            observedDescriptors = null;
            matcher             = null;
            result              = null;
        }
    }
}


