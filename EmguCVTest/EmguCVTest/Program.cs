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
    public class ARObject
    {
        public Dictionary<KeyFrame,Point> ContainerKeyFrames { get; set; }
        public object Obj { get; set; }
    }
    public class KeyFrame
    {
        public Image<Bgr, Byte> Frame       { get; set; }
        public VectorOfKeyPoint KeyPoints   { get; set; }
        public Mat              Descriptors { get; set; }
        public bool             IsContainItem { get; set; }
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

        static int counter = 0;
        [STAThread]
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("Choose a file");
                OpenFileDialog OF = new OpenFileDialog();
                OF.ShowDialog();
                withKeyframes(OF.FileName);
            }
            catch (Exception ex)
            {

            }
        }
        
        private static void withKeyframes(string videoSource)
        {
            ImageViewer viewer = new ImageViewer();
            try
            {
                Capture capture = videoSource != null ? new Capture(videoSource) : new Capture(0);
                Image<Bgr, Byte> result = null;
                KeyFrame keyFrame = null;

                Application.Idle += new EventHandler(delegate (object sender, EventArgs e)
                {
                    counter++;
                    if (keyFrames.Count < 15)
                    {
                        IDrawer drawer = new SURFDrawer();
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
                                if (counter == 65)
                                {
                                    //homography ile camera pozu ayarla, nesneyi koy
                                }
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
    }
}
