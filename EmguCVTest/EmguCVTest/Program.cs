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
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                ImageViewer viewer = new ImageViewer();
                Capture capture = new Capture(0);
                //FastDetector detector = new FastDetector(10);
                Image<Bgr, Byte> lastFrame = null;

                

                Application.Idle += new EventHandler(delegate (object sender, EventArgs e)
                {
                    Image<Bgr, Byte> result = null;
                    FastDrawer drawer = new FastDrawer();
                    try
                    {
                        Mat frame = capture.QueryFrame();
                        result = frame.ToImage<Bgr, Byte>();
                        //VectorOfKeyPoint keypoints = new VectorOfKeyPoint(detector.Detect(result));
                        //Features2DToolbox.DrawKeypoints(result, keypoints, result, new Bgr(System.Drawing.Color.Red), Features2DToolbox.KeypointDrawType.Default);
                        if (lastFrame != null)
                            result = drawer.Draw(lastFrame, result);
                        lastFrame = frame.ToImage<Bgr, Byte>();
                        viewer.Image = result;
                        //result.Dispose();
                    }
                    catch (Exception ex1)
                    {

                    }
                    finally
                    {
                        result.Dispose();
                        drawer.Dispose();
                    }
                });
                    viewer.ShowDialog(); 

            }
            catch (Exception ex)
            {

            }
        }

        

        //public static Image<Bgr, Byte> Draw(Image<Bgr, Byte> modelImage, Image<Bgr, Byte> observedImage)
        //{
        //    Mat homography = null;

        //    FastDetector fastCPU = new FastDetector(15);
        //    VectorOfKeyPoint modelKeyPoints;
        //    VectorOfKeyPoint observedKeyPoints;
        //    VectorOfVectorOfDMatch indices = new VectorOfVectorOfDMatch();

        //    BriefDescriptorExtractor descriptor = new BriefDescriptorExtractor();

        //    Mat mask;
        //    int k = 2;
        //    double uniquenessThreshold = 0.8;

        //    //extract features from the object image
        //    modelKeyPoints = new VectorOfKeyPoint(fastCPU.Detect(modelImage));
        //    Mat modelDescriptors = new Mat();
        //    descriptor.Compute(modelImage, modelKeyPoints, modelDescriptors);
            
        //    // extract features from the observed image
        //    observedKeyPoints = new VectorOfKeyPoint(fastCPU.Detect(observedImage));
        //    Mat observedDescriptors = new Mat();
        //    descriptor.Compute(observedImage, observedKeyPoints, observedDescriptors);

        //    if (modelKeyPoints.Size == 0 || observedKeyPoints.Size == 0)
        //        return observedImage;
        //    BFMatcher matcher = new BFMatcher(DistanceType.L2);
        //    matcher.Add(modelDescriptors);
            
           
        //    matcher.KnnMatch(observedDescriptors, indices, k, null);
        //    mask = new Mat(observedDescriptors.Size, observedDescriptors.Depth, observedDescriptors.NumberOfChannels);
        //    mask.SetTo(new MCvScalar(255));
        //    Features2DToolbox.VoteForUniqueness(indices, uniquenessThreshold, mask);


        //    int nonZeroCount = CvInvoke.CountNonZero(mask);
        //    if (nonZeroCount >= 4)
        //    {
        //        nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, indices, mask, 1.5, 20);
        //        if (nonZeroCount >= 4)
        //            homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints, observedKeyPoints, indices, mask, 2);
        //    }

        //    //Draw the matched keypoints
        //    Mat result = new Mat();
        //    Features2DToolbox.DrawMatches(modelImage, modelKeyPoints, observedImage, observedKeyPoints,
        //    indices, result, new MCvScalar(255,200,150), new MCvScalar(0, 0, 255),null, Features2DToolbox.KeypointDrawType.DrawRichKeypoints);
        //    Image<Bgr, byte> res = result.ToImage<Bgr, byte>();
        //    #region draw the projected region on the image
        //    if (homography != null)
        //    {  //draw a rectangle along the projected model
        //        Rectangle rect = res.ROI;
        //        PointF[] pts = new PointF[] {
        //        new PointF(rect.Left, rect.Bottom),
        //        new PointF(rect.Right, rect.Bottom),
        //        new PointF(rect.Right, rect.Top),
        //        new PointF(rect.Left, rect.Top)};

        //        //CvInvoke.ProjectPoints(pts , homography);
        //        //homography.ProjectPoints(pts);

        //        res.DrawPolyline(Array.ConvertAll<PointF, Point>(pts, Point.Round), true, new Bgr(Color.Red), 5);
        //    }
        //    #endregion
        //    modelImage.Dispose();
        //    return res;
        //}

    }
}
