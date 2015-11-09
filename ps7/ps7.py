"""Problem Set 7: Particle Filter Tracking."""

import numpy as np
import cv2
from random import randint
import math

import os

# I/O directories
input_dir = "input"
output_dir = "output"
should_print = True


# Assignment code
class ParticleFilter(object):
    """A particle filter tracker, encapsulating state, initialization and update methods."""

    def __init__(self, frame, template, **kwargs):
        """Initialize particle filter object.

        Parameters
        ----------
            frame: color BGR uint8 image of initial video frame, values in [0, 255]
            template: color BGR uint8 image of patch to track, values in [0, 255]
            kwargs: keyword arguments needed by particle filter model, including:
            - num_particles: number of particles
        """
        self.num_particles = kwargs.get('num_particles', 100)  # extract num_particles (default: 100)
        # TODO: Your code here - extract any additional keyword arguments you need and initialize state
        self.sigma = kwargs.get('sigma', 10)

        self.template = template

        #particles - x,y pairs
        self.particles = []


        #weights - same indicies as the particles (e.g. weight[3] applies to particles[3])
        #init weights to be uniform
        self.weights = np.ones(self.num_particles, dtype=np.float) / self.num_particles

        start_near_temp = kwargs.get('start_near_temp', True)
        buf = 30
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        for i in range(0,self.num_particles):
            #select a random (x,y)
            if start_near_temp:
                self.particles.append((randint(kwargs.get('x') - buf, kwargs.get('x') + kwargs.get('w') + buf),
                                       randint(kwargs.get('y') - buf, kwargs.get('y') + kwargs.get('h') + buf)))
            else:
                # randint uses the endpoints so subtract 1 from the end
                self.particles.append((randint(0, frame_width - 1), randint(0, frame_height - 1)))


    def calc_similarity(self, template, patch, sigma):
        mean_std_err = ((template - patch) ** 2).mean(axis=None).astype(np.float)
        similarity = math.exp(-mean_std_err / (2.0 * (sigma **2)))

        return similarity


    def re_sample(self):
        new_particles = []
        new_weights = []

        # protect ourselves
        self.weights /= sum(self.weights)
        weighted_rand_particles = np.random.multinomial(self.num_particles, self.weights, size=1)[0]

        for i in range(self.num_particles):
            for num_parts in range(weighted_rand_particles[i]):
                new_particles.append(self.particles[i])
                new_weights.append(self.weights[i])

        self.particles = new_particles
        self.weights = new_weights / sum(new_weights)


    def process(self, frame):
        """Process a frame (image) of video and update filter state.

        Parameters
        ----------
            frame: color BGR uint8 image of current video frame, values in [0, 255]
        """

        #for each particle,
        #get frame centered at that point
        #calc MSE with the template
        #add MSE to all weights by particle
        #track how much added total to normalize
        #create noise1 & noise2 - noise = np.random.normal(mu, sigma, 1)
        #add noise to x, add noise to y
        #normalize all weights by amount added

        self.re_sample()

        amountAdded = 0.0
        for i in range(0, self.num_particles):
            # if should_print : print "particles", self.particles[i]
            patch = get_patch(frame, self.particles[i], self.template.shape)

            # ignore patches at the edges of the image
            if patch.shape == self.template.shape:

                similarity = self.calc_similarity(self.template, patch, self.sigma)

                self.weights[i] += similarity
                amountAdded += similarity
                noise0 = np.random.normal(0, self.sigma, 1)
                noise1 = np.random.normal(0, self.sigma, 1)

                self.particles[i] = (int(self.particles[i][0] + noise0), int(self.particles[i][1] + noise1))


        if amountAdded > 0:
            self.weights /= amountAdded
            self.weights /= sum(self.weights)

        pass  # TODO: Your code here - use the frame as a new observation (measurement) and update model

    def render(self, frame_out):
        """Visualize current particle filter state.

        Parameters
        ----------
            frame_out: copy of frame to overlay visualization on
        """
        # Note: This may not be called for all frames, so don't do any model updates here!
        # TODO: Your code here - draw particles, tracking window and a circle to indicate spread
        u_weighted_mean = 0
        v_weighted_mean = 0
        for i in range(self.num_particles):
            u = self.particles[i][0]
            v = self.particles[i][1]
            cv2.circle(frame_out, (int(u), int(v)), 1, (0, 0, 255))
            u_weighted_mean += u * self.weights[i]
            v_weighted_mean += v * self.weights[i]

        sum_dist = 0
        for i in range(self.num_particles):
            part_pt = self.particles[i]
            u = part_pt[0]
            v = part_pt[1]
            sum_dist += math.sqrt((u - u_weighted_mean)**2 + (v - v_weighted_mean)**2)

        radius = int(sum_dist / self.num_particles)
        center = (int(u_weighted_mean), int(v_weighted_mean))
        x, y, h, w = get_rect(center, self.template.shape)

        cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 255, 0))
        cv2.circle(frame_out, center, radius, (0, 255, 0))


def get_patch(frame, particle, shape_needed):
    x, y, h, w = get_rect(particle, shape_needed)

    # if should_print:
    #     print "y,h,x,w", y, h, x, w

    return frame[y:y + h, x:x + w]


def get_rect(point, shape_needed):
    h = int(shape_needed[0])
    w = int(shape_needed[1])
    x = int(point[0] - (w / 2))
    y = int(point[1] - (h / 2))

    return x, y, h, w

class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker that updates its appearance model over time."""

    def __init__(self, frame, template, **kwargs):
        """Initialize appearance model particle filter object (parameters same as ParticleFilter)."""
        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor
        # TODO: Your code here - additional initialization steps, keyword arguments
        self.alpha = kwargs.get('kwargs', 0.3)

    # TODO: Override process() to implement appearance model update
    def process(self, frame):
        alpha = self.alpha
        amountAdded = 0.0
        best_patch = []
        max_sim = 0
        for i in range(0, self.num_particles):
            # if should_print : print "particles", self.particles[i]
            patch = get_patch(frame, self.particles[i], self.template.shape)

            # ignore patches at the edges of the image
            if patch.shape == self.template.shape:

                similarity = self.calc_similarity(self.template, patch, self.sigma)
                if similarity > max_sim:
                    best_patch = patch
                    max_sim = similarity

                self.weights[i] += similarity
                amountAdded += similarity
                noise0 = np.random.normal(0, self.sigma, 1)
                noise1 = np.random.normal(0, self.sigma, 1)

                self.particles[i] = (int(self.particles[i][0] + noise0), int(self.particles[i][1] + noise1))

        self.template = alpha * best_patch + (1 - alpha) * self.template

        if amountAdded > 0:
            self.weights /= amountAdded
            self.weights /= sum(self.weights)

        self.re_sample()

    # TODO: Override render() if desired (shouldn't have to, ideally)


class HistogramPF(ParticleFilter):
    """A variation of particle filter tracker that updates its appearance model over time."""

    def __init__(self, frame, template, **kwargs):
        """Initialize appearance model particle filter object (parameters same as ParticleFilter)."""
        super(HistogramPF, self).__init__(frame, template, **kwargs)  # call base class constructor

    def calc_hist(self, image):
        hist = cv2.calcHist([image.astype(np.uint8)], [0], None, [8],[0, 255])
        hist = cv2.normalize(hist).flatten()

        return hist

    def calc_similarity(self, template, patch, sigma):
        template_hist = self.calc_hist(template)
        patch_hist = self.calc_hist(patch)
        # smaller val indicates more similar for chi-squared. So, flip it
        # sim = cv2.compareHist(template_hist, patch_hist, cv2.cv.CV_COMP_CHISQR)
        # similarity = math.exp(-sim / (2.0 * (sigma **2)))
        similarity = 1.0 / cv2.compareHist(template_hist, patch_hist, cv2.cv.CV_COMP_CHISQR)

        # sim = cv2.compareHist(template_hist, patch_hist, cv2.cv.CV_COMP_CORREL)

        return similarity





# Driver/helper code
def get_template_rect(rect_filename):
    """Read rectangular template bounds from given file.

    The file must define 4 numbers (floating-point or integer), separated by whitespace:
    <x> <y>
    <w> <h>

    Parameters
    ----------
        rect_filename: path to file defining template rectangle

    Returns
    -------
        template_rect: dictionary specifying template bounds (x, y, w, h), as float or int

    """
    with open(rect_filename, 'r') as f:
        values = [float(v) for v in f.read().split()]
        return dict(zip(['x', 'y', 'w', 'h'], values[0:4]))


def run_particle_filter(pf_class, video_filename, template_rect, save_frames={}, **kwargs):
    """Instantiate and run a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any keyword arguments.

    Parameters
    ----------
        pf_class: particle filter class to instantiate (e.g. ParticleFilter)
        video_filename: path to input video file
        template_rect: dictionary specifying template bounds (x, y, w, h), as float or int
        save_frames: dictionary of frames to save {<frame number>|'template': <filename>}
        kwargs: arbitrary keyword arguments passed on to particle filter class
    """
    # Open video file
    video = cv2.VideoCapture(video_filename)

    # Initialize objects
    template = None
    pf = None
    frame_num = 0

    fps = 60
    #capSize = gray.shape # this is the size of my source video
    size = (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
    vout = cv2.VideoWriter()
    success = vout.open('output/output.mov',fourcc,fps,size,True)

    # Loop over video (till last frame or Ctrl+C is pressed)
    while True:
        try:
            # Try to read a frame
            okay, frame = video.read()
            if not okay:
                print "done"
                break  # no more frames, or can't read video

            color_frame = frame.copy()
            frame = create_simple_frame(frame)

            # Extract template and initialize (one-time only)
            if template is None:
                y = int(template_rect['y'])
                x = int(template_rect['x'])
                h = int(template_rect['h'])
                w = int(template_rect['w'])

                kwargs['x'] = x
                kwargs['y'] = y
                kwargs['h'] = h
                kwargs['w'] = w

                template = frame[y:y + h, x:x + w]

                if 'template' in save_frames:
                    cv2.imwrite(save_frames['template'], template)

                pf = pf_class(frame, template, **kwargs)

            # Process frame
            pf.process(frame)  # TODO: implement this!

            pf.render(color_frame)
            vout.write(color_frame)

            # Render and save output, if indicated
            if kwargs['show_img']:
                # if count == 140:
                if (frame_num % 10) == 0:
                    # pf.render(color_frame)
                    cv2.imshow('num parts (' + str(kwargs['num_particles']) +') sigma (' + str(kwargs['sigma']) + ') Frame: ' + str(frame_num), color_frame)
                    if frame_num > 0:
                        cv2.destroyWindow('num parts (' + str(kwargs['num_particles']) +') sigma (' + str(kwargs['sigma']) + ') Frame: ' + str(frame_num - 1))
            else:
                # if frame_num == 15:
                #     cv2.imwrite("output/frame.png", color_frame)
                #     exit()
                if frame_num in save_frames:
                    # pf.render(color_frame)
                    cv2.imwrite(save_frames[frame_num], color_frame)


            # Update frame number
            frame_num += 1
        except KeyboardInterrupt:  # press ^C to quit
            break

def create_simple_frame(frame):
    weighted = False

    if weighted:
        # Weighted vals
        b, g, r = cv2.split(frame)
        frame = (b * 0.3) + (g * 0.58) + (r * 0.12)
    else:
        # Gray
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float)


    return frame

def one_a(show_img=False):
    run_particle_filter(ParticleFilter,  # particle filter model class
        os.path.join(input_dir, "pres_debate.avi"),  # input video
        get_template_rect(os.path.join(input_dir, "pres_debate.txt")),  # suggested template window (dict)
        # Note: To specify your own window, directly pass in a dict: {'x': x, 'y': y, 'w': width, 'h': height}
        {
            'template': os.path.join(output_dir, 'ps7-1-a-1.png'),
            28: os.path.join(output_dir, 'ps7-1-a-2.png'),
            84: os.path.join(output_dir, 'ps7-1-a-3.png'),
            144: os.path.join(output_dir, 'ps7-1-a-4.png')
        },  # frames to save, mapped to filenames, and 'template' if desired
        num_particles=200, sigma=10,  measurement_noise=0.05, show_img=show_img, start_near_temp=True)  # TODO: specify other keyword args that your model expects, e.g. measurement_noise=0.2


def one_b(show_img=True):
    # 1b
    # TODO: Repeat 1a, but vary template window size and discuss trade-offs (no output images required)
    rect = {'y': 175.0, 'x': 320.0, 'w': 103.0, 'h': 129.0}
    rect['y'] -= 60
    rect['x'] -= 60
    rect['w'] += 60
    rect['h'] += 60
    run_particle_filter(ParticleFilter,  # particle filter model class
        os.path.join(input_dir, "pres_debate.avi"),  # input video
        rect,  # suggested template window (dict)
        # Note: To specify your own window, directly pass in a dict: {'x': x, 'y': y, 'w': width, 'h': height}
        {},  # frames to save, mapped to filenames, and 'template' if desired
        num_particles=200, sigma=10,  measurement_noise=0.05, show_img=show_img, start_near_temp=True)  # TODO: specify other keyword args that your model expects, e.g. measurement_noise=0.2


def one_c(show_img=True):
    # 1c
    # TODO: Repeat 1a, but vary the sigma_MSE parameter (no output images required)
    for sigma in range(20, 0, -5):
        run_particle_filter(ParticleFilter,  # particle filter model class
            os.path.join(input_dir, "pres_debate.avi"),  # input video
            get_template_rect(os.path.join(input_dir, "pres_debate.txt")),  # suggested template window (dict)
            # Note: To specify your own window, directly pass in a dict: {'x': x, 'y': y, 'w': width, 'h': height}
            {},  # frames to save, mapped to filenames, and 'template' if desired
            num_particles=200, sigma=sigma,  measurement_noise=0.05, show_img=show_img, start_near_temp=True)  # TODO: specify other keyword args that your model expects, e.g. measurement_noise=0.2


def one_d(show_img=True):
    # 1d
    # TODO: Repeat 1a, but try to optimize (minimize) num_particles (no output images required)
    for num_particles in range(25, 125, 10):
        run_particle_filter(ParticleFilter,  # particle filter model class
            os.path.join(input_dir, "pres_debate.avi"),  # input video
            get_template_rect(os.path.join(input_dir, "pres_debate.txt")),  # suggested template window (dict)
            # Note: To specify your own window, directly pass in a dict: {'x': x, 'y': y, 'w': width, 'h': height}
            {},  # frames to save, mapped to filenames, and 'template' if desired
            num_particles=num_particles, sigma=15,  measurement_noise=0.05, show_img=show_img, start_near_temp=False)  # TODO: specify other keyword args that your model expects, e.g. measurement_noise=0.2


def one_e(show_img=False):
    run_particle_filter(ParticleFilter,
                        os.path.join(input_dir, "noisy_debate.avi"),
                        get_template_rect(os.path.join(input_dir, "noisy_debate.txt")),
                        {
                            14: os.path.join(output_dir, 'ps7-1-e-1.png'),
                            32: os.path.join(output_dir, 'ps7-1-e-2.png'),
                            46: os.path.join(output_dir, 'ps7-1-e-3.png')
                        },
                        num_particles=500, sigma=15,  measurement_noise=0.1, show_img=show_img, start_near_temp=False)  # TODO: Tune parameters so that model can continuing tracking through noise


def two_a(show_img=False):
    # TODO: Implement AppearanceModelPF (derived from ParticleFilter)
    # TODO: Run it on pres_debate.avi to track Romney's left hand, tweak parameters to track up to frame 140
    run_particle_filter(AppearanceModelPF,
                        os.path.join(input_dir, "pres_debate.avi"),
                        get_template_rect(os.path.join(input_dir, "hand.txt")),
                        {
                            'template': os.path.join(output_dir, 'ps7-2-a-1.png'),
                            15: os.path.join(output_dir, 'ps7-2-a-2.png'),
                            50: os.path.join(output_dir, 'ps7-2-a-3.png'),
                            140: os.path.join(output_dir, 'ps7-2-a-4.png')
                        },
                        num_particles=500, sigma=15, measurement_noise=0.01, show_img=show_img, start_near_temp=True)


def two_b(show_img=False):
    # TODO: Run AppearanceModelPF on noisy_debate.avi, tweak parameters to track hand up to frame 140
    # for i in range(5):
    #     run_particle_filter(AppearanceModelPF,
    #                     os.path.join(input_dir, "noisy_debate.avi"),
    #                     get_template_rect(os.path.join(input_dir, "hand.txt")),
    #                     {
    #                         'template': os.path.join(output_dir, 'ps7-2-b-1.png'),
    #                         15: os.path.join(output_dir, "2b/" + str(i) + '-ps7-2-b-2.png'),
    #                         50: os.path.join(output_dir, "2b/" + str(i) + '-ps7-2-b-3.png'),
    #                         140: os.path.join(output_dir, "2b/" + str(i) + '-ps7-2-b-4.png')
    #                     },
    #                     num_particles=300, sigma=25, measurement_noise=0.8, show_img=show_img, start_near_temp=True)

    run_particle_filter(AppearanceModelPF,
                        os.path.join(input_dir, "noisy_debate.avi"),
                        get_template_rect(os.path.join(input_dir, "fuzzy_hand.txt")),
                        {
                            'template': os.path.join(output_dir, 'ps7-2-b-1.png'),
                            15: os.path.join(output_dir, 'ps7-2-b-2.png'),
                            50: os.path.join(output_dir, 'ps7-2-b-3.png'),
                            140: os.path.join(output_dir, 'ps7-2-b-4.png')
                        },
                        num_particles=500, sigma=10, measurement_noise=0.2, alpha=0.5, show_img=show_img, start_near_temp=True)

    # rect = get_template_rect(os.path.join(input_dir, "hand.txt"))
    # for i in range(1, 15):
    #     sigma = int(i)
    #     for j in range(1, 100, 5):
    #         noise = j / 100.0
    #         run_particle_filter(AppearanceModelPF,
    #                     os.path.join(input_dir, "noisy_debate.avi"),
    #                     rect,
    #                     {
    #                         140: os.path.join(output_dir, '2b/sigma' + str(sigma) + '_noise' + str(noise) + '.png')
    #                     },
    #                     num_particles=500, sigma=sigma, measurement_noise=0.025, show_img=False, start_near_temp=True)

def three_a(show_img=False):
    run_particle_filter(HistogramPF,
                        os.path.join(input_dir, "pres_debate.avi"),  # input video
                        get_template_rect(os.path.join(input_dir, "pres_debate.txt")),  # suggested template window (dict)
                        # Note: To specify your own window, directly pass in a dict: {'x': x, 'y': y, 'w': width, 'h': height}
                        {
                            'template': os.path.join(output_dir, 'ps7-3-a-1.png'),
                            28: os.path.join(output_dir, 'ps7-3-a-2.png'),
                            84: os.path.join(output_dir, 'ps7-3-a-3.png'),
                            144: os.path.join(output_dir, 'ps7-3-a-4.png')
                        },  # frames to save, mapped to filenames, and 'template' if desired
                        num_particles=200, sigma=20,  measurement_noise=0.05, show_img=show_img, start_near_temp=False)

def three_b(show_img=False):
    run_particle_filter(HistogramPF,
                        os.path.join(input_dir, "noisy_debate.avi"),
                        get_template_rect(os.path.join(input_dir, "fuzzy_hand.txt")),
                        {
                            'template': os.path.join(output_dir, 'ps7-3-b-1.png'),
                            15: os.path.join(output_dir, 'ps7-3-b-2.png'),
                            50: os.path.join(output_dir, 'ps7-3-b-3.png'),
                            140: os.path.join(output_dir, 'ps7-3-b-4.png')
                        },
                        num_particles=400, sigma=3, measurement_noise=1.5, show_img=show_img, start_near_temp=True)


def main():
    """ Note: Comment out parts of this code as necessary"""
    """ 1a """
    one_a()

    """ 1b """
    one_b()

    """ 1c """
    one_c()

    """ 1d """
    one_d()

    """ 1e """
    one_e()

    """ 2a """
    two_a()

    """ 2b """
    two_b()

    # EXTRA CREDIT
    """ 3: Use color histogram distance instead of MSE (you can implement a derived class similar to AppearanceModelPF) """
    three_a()

    three_b()


    # 4: Implement a more sophisticated model to deal with occlusions and size/perspective changes


if __name__ == "__main__":
    main()