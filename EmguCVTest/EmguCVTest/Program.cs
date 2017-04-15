using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.CV.Util;
using Emgu.CV.XFeatures2D;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;

namespace EmguCVTest
{
    class Program
    {
        public class KeyFrame
        {
            public Image<Bgr, Byte> Frame      { get; set; }
            public VectorOfKeyPoint KeyPoints  { get; set; }
        }
        static void Main(string[] args)
        {
            try
            {
                ImageViewer viewer = new ImageViewer();
                Capture capture = new Capture(0);
                //FastDetector detector = new FastDetector(10);
                Image<Bgr, Byte> lastFrame = null;
                
                List<KeyFrame> keyFrames = new List<KeyFrame>();
                Application.Idle += new EventHandler(delegate (object sender, EventArgs e)
                {
                    Image<Bgr, Byte> result = null;
                    //FastDrawer drawer = new FastDrawer();
                    if (keyFrames.Count < 45)
                        try
                        {
                            Mat frame = capture.QueryFrame();
                            result = frame.ToImage<Bgr, Byte>();
                            //VectorOfKeyPoint keypoints = new VectorOfKeyPoint(detector.Detect(result));
                            //Features2DToolbox.DrawKeypoints(result, keypoints, result, new Bgr(System.Drawing.Color.Red), Features2DToolbox.KeypointDrawType.Default);
                            if (lastFrame != null)
                                result = /*drawer.*/Draw(lastFrame, result, keyFrames);
                            else
                                keyFrames.Add(new KeyFrame() { Frame = frame.ToImage<Bgr, Byte>() });

                            lastFrame = keyFrames.LastOrDefault().Frame;/* frame.ToImage<Bgr, Byte>();*/
                            viewer.Width = frame.Width * 2 + 50;
                            viewer.Image = result;
                            //result.Dispose();
                        }
                        catch (Exception ex1)
                        {

                        }
                        finally
                        {
                            result.Dispose();
                            //drawer.Dispose();
                        }
                    else
                    {
                        //Image<Bgr, Byte> _newImage = keyFrames[0].Frame;
                        //for (int i = 1; i < keyFrames.Count; i++)
                        //{
                        //    _newImage = newImage(_newImage, keyFrames[i].Frame);
                        //}
                        //viewer.Size = _newImage.Size;
                        //viewer.Image = _newImage;
                        keyFrames.RemoveRange(0, keyFrames.Count/2);
                    }
                });
                    viewer.ShowDialog(); 

            }
            catch (Exception ex)
            {

            }
        }



        public static Image<Bgr, Byte> Draw(Image<Bgr, Byte> modelImage, Image<Bgr, Byte> observedImage, List<KeyFrame> keyFrames)
        {
            Mat homography = null;

            FastDetector fastCPU = new FastDetector(25);
            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;
            VectorOfVectorOfDMatch indices = new VectorOfVectorOfDMatch();

            BriefDescriptorExtractor descriptor = new BriefDescriptorExtractor();

            Mat mask;
            int k = 2;
            double uniquenessThreshold = 0.8;

            //extract features from the object image
            modelKeyPoints = new VectorOfKeyPoint(fastCPU.Detect(modelImage));
            Mat modelDescriptors = new Mat();
            descriptor.Compute(modelImage, modelKeyPoints, modelDescriptors);

            // extract features from the observed image
            observedKeyPoints = new VectorOfKeyPoint(fastCPU.Detect(observedImage));
            Mat observedDescriptors = new Mat();
            descriptor.Compute(observedImage, observedKeyPoints, observedDescriptors);
            KeyFrame kf = keyFrames.Where(x => x.KeyPoints == observedKeyPoints).FirstOrDefault();
            if (modelKeyPoints.Size == 0 && observedKeyPoints.Size > 3 && kf == null)
                keyFrames.Add(new KeyFrame() { Frame = observedImage.Clone(), KeyPoints = observedKeyPoints });
            else if(kf != null)
                modelImage = kf.Frame;
            if (modelKeyPoints.Size == 0 || observedKeyPoints.Size == 0)
                return observedImage;
            BFMatcher matcher = new BFMatcher(DistanceType.L2);
            matcher.Add(modelDescriptors);


            matcher.KnnMatch(observedDescriptors, indices, k, null);
            mask = new Mat(observedDescriptors.Size, observedDescriptors.Depth, observedDescriptors.NumberOfChannels);
            mask.SetTo(new MCvScalar(255));
            Features2DToolbox.VoteForUniqueness(indices, uniquenessThreshold, mask);


            int nonZeroCount = CvInvoke.CountNonZero(mask);
            if (nonZeroCount >= 5)
            {
                nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, indices, mask, 1.5, 20);
                if (nonZeroCount >= 5)
                    homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints, observedKeyPoints, indices, mask, 2);
                else
                    keyFrames.Add(new KeyFrame() { Frame = observedImage.Clone(), KeyPoints = observedKeyPoints });
            }

            //Draw the matched keypoints
            Mat result = new Mat();
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
            //modelImage.Dispose();
            return res;
        }


        private static Image<Bgr, Byte> newImage(Image<Bgr, Byte> image1, Image<Bgr, Byte> image2)
        {
            int ImageWidth = 0;
            int ImageHeight = 0;

            //get max width
            if (image1.Width > image2.Width)
                ImageWidth = image1.Width;
            else
                ImageWidth = image2.Width;

            //calculate new height
            ImageHeight = image1.Height + image2.Height;

            //declare new image (large image).
            Image<Bgr, Byte> imageResult;

            Bitmap bitmap = new Bitmap(Math.Max(image1.Width, image2.Width), image1.Height + image2.Height);
            using (Graphics g = Graphics.FromImage(bitmap))
            {
                g.DrawImage(image1.Bitmap, 0, 0);
                g.DrawImage(image2.Bitmap, 0, image1.Height);

            }

            imageResult = new Image<Bgr, Byte>(bitmap);



            return imageResult/*.Resize(0.60, interpolationType: Emgu.CV.CvEnum.Inter.Area)*/;
        }
    }
}
