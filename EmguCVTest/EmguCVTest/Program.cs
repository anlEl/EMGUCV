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
using System.Windows.Forms;
namespace EmguCVTest
{
    public class KeyFrame
    {
        public Image<Bgr, Byte> Frame       { get; set; }
        public VectorOfKeyPoint KeyPoints   { get; set; }
        public Mat              Descriptors { get; set; }
        public PointF[]         GetKeypointPoints()
        {
            PointF[] arr = new PointF[KeyPoints.Size];
            for (int i = 0; i < KeyPoints.Size; i++)
                arr[i] = KeyPoints[i].Point;
            return arr;
        }
    }
    class Program
    {
        static List<KeyFrame> keyFrames = new List<KeyFrame>();
        [STAThread]
        static void Main(string[] args)
        {
            try
            {
                choose:
                Console.WriteLine("For camera press 1, for video press 2");
                int choose = -1;
                int.TryParse(Console.ReadLine(), out choose);
                Console.WriteLine("Without Keyframes press 1, With Keyframes press 2");
                int chooseKF = -1;
                int.TryParse(Console.ReadLine(), out chooseKF);
                if (choose == 1)
                {
                    if (chooseKF == 1)
                        withoutKeyframes(true);
                    else if (chooseKF == 2)
                        withKeyframes(false);
                    else
                        goto choose;
                }
                else if (choose == 2)
                {
                    OpenFileDialog OF = new OpenFileDialog();
                    OF.ShowDialog();
                    if (chooseKF == 1)
                        withoutKeyframes(true, OF.FileName);
                    else if (chooseKF == 2)
                        withKeyframes(false, OF.FileName);
                    else
                        goto choose;
                }
                else
                    goto choose;
                //withKeyframes();
            }
            catch (Exception ex)
            {

            }
        }
        private static void withoutKeyframes(bool isFast,string videoSource = null)
        {
            ImageViewer viewer = new ImageViewer();
            try
            {
                Capture capture     = videoSource != null ? new Capture(videoSource) : new Capture(0);
                Image<Bgr, Byte> lastFrame = null, result  = null;
                Application.Idle   += new EventHandler(delegate (object sender, EventArgs e)
                {
                    IDrawer drawer;
                    if (isFast)
                        drawer = new FastDrawer();
                    else
                        drawer = new SURFDrawer();
                    try
                    {
                        Mat frame = capture.QueryFrame();
                        if (frame.Width > 720 || frame.Height > 480)
                        {
                            double width = 720.0 / frame.Width;
                            double height = 480.0 / frame.Height;

                            CvInvoke.Resize(frame, frame, new Size(), width, height, Inter.Linear);
                        }
                        result = frame.ToImage<Bgr,Byte>();
                        if (lastFrame != null)
                            result = drawer.FindMatch(frame.ToImage<Bgr, Byte>(), lastFrame).Draw(frame.ToImage<Bgr,Byte>(), lastFrame);

                        lastFrame    = frame.ToImage<Bgr, Byte>();
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
                        drawer.Dispose();
                    }
                });

            }
            catch (Exception ex)
            {

            }
            viewer.ShowDialog();
        }
        private static void withKeyframes(bool isFast, string videoSource = null)
        {
            ImageViewer viewer = new ImageViewer();
            try
            {
                Capture capture = videoSource != null ? new Capture(videoSource) : new Capture(0);
                Image<Bgr, Byte> result = null;
                KeyFrame keyFrame = null;

                Application.Idle += new EventHandler(delegate (object sender, EventArgs e)
                {
                    if (keyFrames.Count < 15)
                    {
                        IDrawer drawer;
                        if (isFast)
                            drawer = new FastDrawer();
                        else
                            drawer = new SURFDrawer();
                        try
                        {
                            Mat frame = capture.QueryFrame();
                            if (frame.Width > 720 || frame.Height > 480)
                            {
                                double width = 720.0 / frame.Width;
                                double height = 480.0 / frame.Height;

                                CvInvoke.Resize(frame, frame, new Size(), width, height, Inter.Linear);
                            }
                            Image<Bgr, Byte> framebuffer = frame.ToImage<Bgr, Byte>();
                            if (keyFrames.Count == 0)
                                keyFrames.Add(new KeyFrame() { Frame = framebuffer });
                            for (int i = keyFrames.Count-1; i >=0; i--)
                            {
                                KeyFrame kf = keyFrames[i];
                                drawer.FindMatch(kf, framebuffer, keyFrames);
                                if (drawer.homography != null)
                                {
                                    KeyFrame buffer_kf = keyFrames[keyFrames.Count - 1];
                                    if (kf != buffer_kf)
                                    {
                                        keyFrames[i] = buffer_kf;
                                        keyFrames[keyFrames.Count - 1] = kf;
                                    }
                                    keyFrame = kf;
                                    break;
                                }
                                if (i == 0)
                                {
                                    if (drawer.homography == null)
                                        keyFrames.Add(new KeyFrame() { Frame = framebuffer, KeyPoints = drawer.observedKeyPoints, Descriptors = drawer.observedDescriptors });
                                }
                                drawer.Clear();
                            }
                            result = frame.ToImage<Bgr, Byte>();
                            if(keyFrame != null)
                                result = drawer.Draw(keyFrame, framebuffer);
                            
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
                            drawer.Dispose();
                        }
                    }
                    else
                    {
                        keyFrames.RemoveRange(0, keyFrames.Count / 2);
                    }
                });
            }
            catch
            {

            }
            viewer.ShowDialog();
        }

        public static Image<Bgr, Byte> Draw(Image<Bgr, Byte> modelImage, Image<Bgr, Byte> observedImage, List<KeyFrame> keyFrames)
        {
            Mat homography = null;

            FastDetector fastCPU = new FastDetector(55);
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

            if (modelKeyPoints.Size == 0 || observedKeyPoints.Size == 0)
                return observedImage;
            BFMatcher matcher = new BFMatcher(DistanceType.L2);
            matcher.Add(modelDescriptors);


            matcher.KnnMatch(observedDescriptors, indices, k, null);
            mask = new Mat(observedDescriptors.Size, observedDescriptors.Depth, observedDescriptors.NumberOfChannels);
            mask.SetTo(new MCvScalar(0));
            Features2DToolbox.VoteForUniqueness(indices, uniquenessThreshold, mask);


            int nonZeroCount = CvInvoke.CountNonZero(mask);
            if (nonZeroCount >= 4)
            {
                nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, indices, mask, 1.5, 20);
                if (nonZeroCount >= 4)
                    homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints, observedKeyPoints, indices, mask, 2);
                else
                    keyFrames.Add(new KeyFrame() { Frame = observedImage.Clone(), KeyPoints = observedKeyPoints });
            }
            else
                keyFrames.Add(new KeyFrame() { Frame = observedImage.Clone(), KeyPoints = observedKeyPoints });

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
