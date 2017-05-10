using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.CV.Util;
using Emgu.CV.XFeatures2D;
using System;
using System.Drawing;
using System.Windows.Forms;
using Emgu.CV.Flann;
using Emgu.CV.CvEnum;
using System.Collections.Generic;
using System.Linq;

namespace EmguCVTest
{
    interface IDrawer: IDisposable
    {
        Mat                         homography           {get; set;}
        Feature2D                   CPU                  {get; set;}
        VectorOfKeyPoint            modelKeyPoints       {get; set;}
        VectorOfKeyPoint            observedKeyPoints    {get; set;}
        VectorOfVectorOfDMatch      matches              {get; set;}
        BriefDescriptorExtractor    descriptor           {get; set;}
        Mat                         mask                 {get; set;}
        int                         k                    {get; set;}
        double                      hessianThresh        {get; set;}
        double                      uniquenessThreshold  {get; set;}
        Mat                         modelDescriptors     {get; set;}
        Mat                         observedDescriptors  {get; set;}
        BFMatcher                   matcher              {get; set;}
        Mat                         result               {get; set;}
        void                        Clear                ();
        IDrawer                     FindMatch            (Image<Bgr, Byte> modelImage, Image<Bgr, Byte> observedImage, List<KeyFrame> keyframes = null);
        IDrawer                     FindMatch            (KeyFrame keyFrame,           Image<Bgr, Byte> observedImage, List<KeyFrame> keyframes = null);
        Image<Bgr, Byte>            Draw                 (Image<Bgr, Byte> modelImage, Image<Bgr, Byte> observedImage, List<KeyFrame> keyframes = null);
        Image<Bgr, Byte>            Draw                 (KeyFrame keyFrame, Image<Bgr, Byte> observedImage, List<KeyFrame> keyframes = null);
    }
    class SURFDrawer : IDrawer
    {
        public Mat                         homography               {get; set;}
        public Feature2D                   CPU                      {get; set;}
        public VectorOfKeyPoint            modelKeyPoints           {get; set;}
        public VectorOfKeyPoint            observedKeyPoints        {get; set;}
        public VectorOfVectorOfDMatch      matches                  {get; set;}
        public BriefDescriptorExtractor    descriptor               {get; set;}
        public Mat                         mask                     {get; set;}
        public int                         k                        {get; set;}
        public double                      hessianThresh            {get; set;}
        public double                      uniquenessThreshold      {get; set;}
        public Mat                         modelDescriptors         {get; set;}
        public Mat                         observedDescriptors      {get; set;}
        public BFMatcher                   matcher                  {get; set;}
        public Mat                         result                   {get; set;}

        public SURFDrawer()
        {

            k                       = 2;
            uniquenessThreshold     = 0.8;
            hessianThresh           = 300;
            CPU                     = new SURF(hessianThresh);
            matcher                 = new BFMatcher(DistanceType.L2);
            descriptor              = new BriefDescriptorExtractor();
            homography              = null;
            matches                 = new VectorOfVectorOfDMatch();
            modelDescriptors        = new Mat();
            observedDescriptors     = new Mat();
            result                  = new Mat();
        }
        public IDrawer FindMatch(KeyFrame keyFrame, Image<Bgr, Byte> observedImage, List<KeyFrame> keyframes = null)
        {                
            if(keyFrame.KeyPoints == null)
                keyFrame.KeyPoints = new VectorOfKeyPoint(CPU.Detect(keyFrame.Frame));
            if (keyFrame.Descriptors == null)
            {
                keyFrame.Descriptors = new Mat();
                descriptor.Compute(keyFrame.Frame, keyFrame.KeyPoints, keyFrame.Descriptors);
            }

            // extract features from the observed image
            observedKeyPoints = new VectorOfKeyPoint(CPU.Detect(observedImage));
            descriptor.Compute(observedImage, observedKeyPoints, observedDescriptors);
            matcher.Add(keyFrame.Descriptors);

            matcher.KnnMatch(observedDescriptors, matches, k, null);
            mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(255));
            Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

            int nonZeroCount = CvInvoke.CountNonZero(mask);
            if (nonZeroCount >= 4)
            {
                nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(keyFrame.KeyPoints, observedKeyPoints,
                   matches, mask, 1.5, 20);
                if (nonZeroCount >= 4)
                    homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(keyFrame.KeyPoints, observedKeyPoints, matches, mask, 3);

                nonZeroCount = CvInvoke.CountNonZero(mask);

                if (nonZeroCount < 9)
                    homography = null;               
            }

            return this;
        }
        public IDrawer FindMatch(Image<Bgr, Byte> modelImage, Image<Bgr, Byte> observedImage, List<KeyFrame> keyframes = null)
        {
            
            //extract features from the object image
            modelKeyPoints = new VectorOfKeyPoint(CPU.Detect(modelImage));
            descriptor.Compute(modelImage, modelKeyPoints, modelDescriptors);


            // extract features from the observed image
            observedKeyPoints = new VectorOfKeyPoint(CPU.Detect(observedImage));
            descriptor.Compute(observedImage, observedKeyPoints, observedDescriptors);
            matcher.Add(modelDescriptors);

            matcher.KnnMatch(observedDescriptors, matches, k, null);
            mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(255));
            Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

            int nonZeroCount = CvInvoke.CountNonZero(mask);
            if (nonZeroCount >= 4)
            {
                nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints,
                   matches, mask, 1.5, 20);
                if (nonZeroCount >= 4)
                    homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints, observedKeyPoints, matches, mask, 2);

                nonZeroCount = CvInvoke.CountNonZero(mask);

                if (nonZeroCount < 9)
                    homography = null;

                //if (keyframes != null && homography == null)
                //    keyframes.Add(new KeyFrame() { Frame = observedImage, KeyPoints = observedKeyPoints });
            }

            return this;
        }
        public Image<Bgr, Byte> Draw(KeyFrame keyFrame, Image<Bgr, Byte> observedImage, List<KeyFrame> keyframes = null)
        {

            //FindMatch(modelImage, observedImage, keyframes);
            if (matches.Size == mask.Rows)
                Features2DToolbox.DrawMatches(keyFrame.Frame, keyFrame.KeyPoints, observedImage, observedKeyPoints, matches, result, new MCvScalar(255, 255, 255), new MCvScalar(255, 255, 255), mask, Features2DToolbox.KeypointDrawType.NotDrawSinglePoints);
            else
                Features2DToolbox.DrawMatches(keyFrame.Frame, keyFrame.KeyPoints, observedImage, observedKeyPoints, matches, result, new MCvScalar(255, 255, 255), new MCvScalar(255, 255, 255), null, Features2DToolbox.KeypointDrawType.NotDrawSinglePoints);

            #region draw the projected region on the image

            if (homography != null)
            {
                //draw a rectangle along the projected model
                Rectangle rect = new Rectangle(Point.Empty, keyFrame.Frame.Size);
                PointF[] pts = new PointF[]
                {
                    new PointF(rect.Left, rect.Bottom),
                    new PointF(rect.Right, rect.Bottom),
                    new PointF(rect.Right, rect.Top),
                    new PointF(rect.Left, rect.Top)
                };
                //pts = CvInvoke.PerspectiveTransform(pts, homography);

                Point[] points = Array.ConvertAll<PointF, Point>(pts, Point.Round);
                using (VectorOfPoint vp = new VectorOfPoint(points))
                {
                    CvInvoke.Polylines(result, vp, true, new MCvScalar(255, 0, 0, 255), 5);
                }

            }

            #endregion
            //modelImage.Dispose();
            return result.ToImage<Bgr, Byte>();
        }
        public Image<Bgr, Byte> Draw(Image<Bgr, Byte> modelImage, Image<Bgr, Byte> observedImage, List<KeyFrame> keyframes = null)
        {
            
            //FindMatch(modelImage, observedImage, keyframes);
            
            Features2DToolbox.DrawMatches(modelImage, modelKeyPoints, observedImage, observedKeyPoints,
               matches, result, new MCvScalar(255, 255, 255), new MCvScalar(255, 255, 255), mask, Features2DToolbox.KeypointDrawType.NotDrawSinglePoints);
            
            #region draw the projected region on the image

            if (homography != null)
            {
                //draw a rectangle along the projected model
                Rectangle rect = new Rectangle(Point.Empty, modelImage.Size);
                PointF[] pts = new PointF[]
                {
                    new PointF(rect.Left, rect.Bottom),
                    new PointF(rect.Right, rect.Bottom),
                    new PointF(rect.Right, rect.Top),
                    new PointF(rect.Left, rect.Top)
                };
                //pts = CvInvoke.PerspectiveTransform(pts, homography);

                Point[] points = Array.ConvertAll<PointF, Point>(pts, Point.Round);
                using (VectorOfPoint vp = new VectorOfPoint(points))
                {
                    CvInvoke.Polylines(result, vp, true, new MCvScalar(255, 0, 0, 255), 5);
                }

            }

            #endregion
            //modelImage.Dispose();
            return result.ToImage<Bgr,Byte>();            
        }
        public void Clear()
        {
            k                       = 2;
            uniquenessThreshold     = 0.8;
            hessianThresh           = 300;
            CPU                     = new SURF(hessianThresh);
            matcher                 = new BFMatcher(DistanceType.L2);
            descriptor              = new BriefDescriptorExtractor();
            homography              = null;
            matches                 = new VectorOfVectorOfDMatch();
            modelDescriptors        = new Mat();
            observedDescriptors     = new Mat();
            result                  = new Mat();
        }
        public void Dispose()
        {
            k                       = 2;
            uniquenessThreshold     = 0.8;
            hessianThresh           = 300;
            CPU                 = null;
            matcher                 = null;
            descriptor              = null;
            homography              = null;
            matches                 = null;
            modelDescriptors        = null;
            observedDescriptors     = null;
            result                  = null;
        }
    }
}
