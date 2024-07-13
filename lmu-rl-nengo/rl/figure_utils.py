import sys, os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the correct matplotlibrc
mpl.rc_file(os.path.join(os.path.dirname(__file__), 'matplotlibrc'))
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"""
    \renewcommand*\familydefault{\sfdefault}
    \renewcommand{\vec}[1]{\mathbf{#1}}
    \newcommand{\mat}[1]{\mathbf{#1}}
"""

blues = ["#729fcfff", "#3465a4ff", "#204a87ff"][::-1]
reds = ["#ef2929ff", "#cc0000ff", "#a40000ff"][::-1]
greens = ["#8ae234ff", "#73d216ff", "#4e9a06ff"][::-1]
oranges = ["#fcaf3eff", "#f57900ff", "#ce5c00ff"][::-1]
purples = ["#ad7fa8ff", "#75507bff", "#5c3566ff"][::-1]
yellows = ["#fce94fff", "#edd400ff", "#c4a000ff"][::-1]
grays = [
    "#eeeeecff", "#d3d7cfff", "#babdb6ff", "#888a85ff", "#555753ff",
    "#2e3436ff"
][::-1]

def save(fig, filename, suffix=""):
    # Special treatment for PDFs. We need to run the resulting PDF
    # through Ghostscript to
    # a) trim the figures properly
    # b) subset fonts
    target_file, target_ext = os.path.splitext(filename)
    target_file += suffix
    if target_ext == ".pdf":
        target = target_large = target_file + ".large" + target_ext
    else:
        target = target_file + target_ext
    print("Saving to", target)
    fig.savefig(target,
                bbox_inches='tight',
                pad_inches=0.05,
                transparent=True)

    if target_ext == ".pdf":
        import subprocess
        import re

        # Extract the bounding box
        print("Extracting bounding box of file", target_large)
        gs_out = subprocess.check_output(
            ["gs", "-o", "-", "-sDEVICE=bbox", target_large],
            stderr=subprocess.STDOUT)
        pattern = r"^%%HiResBoundingBox:\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
        x0, y0, x1, y1 = map(
            float,
            re.search(pattern, str(gs_out, "utf-8"),
                      re.MULTILINE).groups())

        # Add a small extension to the bounding box
        pad = 0.5
        x0 -= pad
        x1 += pad
        y0 -= pad
        y1 += pad

        # Run ghostscript again to crop the file
        # See https://stackoverflow.com/a/46058965
        target = target_file + target_ext
        print("Optimising PDF and saving to", target)
        subprocess.check_output([
            "gs", "-o", target, "-dEmbedAllFonts=true",
            "-dSubsetFonts=true", "-dCompressFonts=true",
            "-dPDFSETTINGS=/prepress", '-dDoThumbnails=false',
            "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5",
            f"-dDEVICEWIDTHPOINTS={x1 - x0}",
            f"-dDEVICEHEIGHTPOINTS={y1 - y0}", "-dFIXEDMEDIA", "-c",
            f"<</PageOffset [-{x0} -{y0}]>>setpagedevice", "-f",
            target_large
        ])

        # Remove the large temporary file
        os.unlink(target_large)

def outside_ticks(ax):
    ax.tick_params(direction="out", which="both")

def remove_frame(ax):
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)

def add_frame(self, ax):
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_visible(True)

def annotate(ax,
             x0,
             y0,
             x1,
             y1,
             s=None,
             ha="center",
             va="center",
             fontdict=None,
             zorder=None,
             color="k"):
    ax.plot([x0, x1], [y0, y1],
            color=color,
            linewidth=0.5,
            linestyle=(0, (1, 1)),
            clip_on=False,
            zorder=zorder)
    ax.plot([x0], [y0],
            'o',
            color=color,
            markersize=1,
            clip_on=False,
            zorder=zorder)
    if not s is None:
        ax.text(x1,
                y1,
                s,
                ha=ha,
                va=va,
                bbox={
                    "pad": 1.0,
                    "color": "w",
                    "linewidth": 0.0,
                },
                fontdict=fontdict,
                zorder=zorder)


def vslice(ax, x, y0, y1, **kwargs):
    ax.plot([x, x], [y0, y1],
            'k-',
            linewidth=0.75,
            clip_on=False,
            **kwargs)
    ax.plot(x, y0, 'k_', linewidth=0.75, clip_on=False, **kwargs)
    ax.plot(x, y1, 'k_', linewidth=0.75, clip_on=False, **kwargs)


def hslice(ax, x0, x1, y, **kwargs):
    ax.plot([x0, x1], [y, y],
            'k-',
            linewidth=0.75,
            clip_on=False,
            **kwargs)
    ax.plot(x0, y, 'k|', linewidth=0.75, clip_on=False, **kwargs)
    ax.plot(x1, y, 'k|', linewidth=0.75, clip_on=False, **kwargs)


def timeslice(ax, x0, x1, y, **kwargs):
    hslice(ax, x0, x1, y, **kwargs)

# from https://github.com/iruletheworld/matplotlib-curly-brace/blob/master/curlyBrace.py

def getAxSize(fig, ax):
    '''
    .. _getAxSize :

    Get the axes size in pixels.

    Parameters
    ----------
    fig : matplotlib figure object
        The of the target axes.

    ax : matplotlib axes object
        The target axes.

    Returns
    -------
    ax_width : float
        The axes width in pixels.

    ax_height : float
        The axes height in pixels.

    Reference
    -----------
    https://stackoverflow.com/questions/19306510/determine-matplotlib-axis-size-in-pixels
    '''

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_width, ax_height = bbox.width, bbox.height
    ax_width *= fig.dpi
    ax_height *= fig.dpi

    return ax_width, ax_height

def curlyBrace(fig, ax, p1, p2, k_r=0.1, bool_auto=True, str_text='', int_line_num=2, fontdict={}, **kwargs):
# def curlyBrace(fig, ax, p1, p2, k_r=0.1, bool_auto=True, str_text='', int_line_num=2, fontdict={}, **kwargs):
    '''
    .. _curlyBrace :

    Plot an optionally annotated curly bracket on the given axes of the given figure.

    Note that the brackets are anti-clockwise by default. To reverse the text position, swap
    "p1" and "p2".

    Note that, when the axes aspect is not set to "equal", the axes coordinates need to be
    transformed to screen coordinates, otherwise the arcs may not be seeable. 

    Parameters
    ----------
    fig : matplotlib figure object
        The of the target axes.

    ax : matplotlib axes object
        The target axes.

    p1 : two element numeric list
        The coordinates of the starting point.

    p2 : two element numeric list
        The coordinates of the end point.

    k_r : float
        This is the gain controlling how "curvy" and "pointy" (height) the bracket is.

        Note that, if this gain is too big, the bracket would be very strange.

    bool_auto : boolean
        This is a switch controlling wether to use the auto calculation of axes
        scales.

        When the two axes do not have the same aspects, i.e., not "equal" scales,
        this should be turned on, i.e., True.

        When "equal" aspect is used, this should be turned off, i.e., False.

        If you do not set this to False when setting the axes aspect to "equal",
        the bracket will be in funny shape.

        Default = True

    str_text : string
        The annotation text of the bracket. It would displayed at the mid point
        of bracket with the same rotation as the bracket.

        By default, it follows the anti-clockwise convention. To flip it, swap 
        the end point and the starting point.

        The appearance of this string can be set by using "fontdict", which follows
        the same syntax as the normal matplotlib syntax for font dictionary.

        Default = empty string (no annotation)

    int_line_num : int
        This argument determines how many lines the string annotation is from the summit
        of the bracket.

        The distance would be affected by the font size, since it basically just a number of
        lines appended to the given string.

        Default = 2

    fontdict : dictionary
        This is font dictionary setting the string annotation. It is the same as normal
        matplotlib font dictionary.

        Default = empty dict

    **kwargs : matplotlib line setting arguments
        This allows the user to set the line arguments using named arguments that are
        the same as in matplotlib.

    Returns
    -------
    theta : float
        The bracket angle in radians.

    summit : list
        The positions of the bracket summit.

    arc1 : list of lists
        arc1 positions.

    arc2 : list of lists
        arc2 positions.

    arc3 : list of lists
        arc3 positions.

    arc4 : list of lists
        arc4 positions.

    Reference
    ----------
    https://uk.mathworks.com/matlabcentral/fileexchange/38716-curly-brace-annotation
    '''

    pt1 = [None, None]
    pt2 = [None, None]

    ax_width, ax_height = getAxSize(fig, ax)

    ax_xlim = list(ax.get_xlim())
    ax_ylim = list(ax.get_ylim())

    # log scale consideration
    if 'log' in ax.get_xaxis().get_scale():

        if p1[0] > 0.0:

            pt1[0] = np.log(p1[0])

        elif p1[0] < 0.0:

            pt1[0] = -np.log(abs(p1[0]))

        else:

            pt1[0] = 0.0

        if p2[0] > 0.0:

            pt2[0] = np.log(p2[0])

        elif p2[0] < 0.0:

            pt2[0] = -np.log(abs(p2[0]))

        else:

            pt2[0] = 0

        for i in range(0, len(ax_xlim)):

            if ax_xlim[i] > 0.0:

                ax_xlim[i] = np.log(ax_xlim[i])

            elif ax_xlim[i] < 0.0:

                ax_xlim[i] = -np.log(abs(ax_xlim[i]))

            else:

                ax_xlim[i] = 0.0

    else:

        pt1[0] = p1[0]
        pt2[0] = p2[0]

    if 'log' in ax.get_yaxis().get_scale():

        if p1[1] > 0.0:

            pt1[1] = np.log(p1[1])

        elif p1[1] < 0.0:

            pt1[1] = -np.log(abs(p1[1]))

        else:

            pt1[1] = 0.0

        if p2[1] > 0.0:

            pt2[1] = np.log(p2[1])

        elif p2[1] < 0.0:

            pt2[1] = -np.log(abs(p2[1]))

        else:

            pt2[1] = 0.0

        for i in range(0, len(ax_ylim)):

            if ax_ylim[i] > 0.0:

                ax_ylim[i] = np.log(ax_ylim[i])

            elif ax_ylim[i] < 0.0:

                ax_ylim[i] = -np.log(abs(ax_ylim[i]))

            else:

                ax_ylim[i] = 0.0

    else:

        pt1[1] = p1[1]
        pt2[1] = p2[1]

    # get the ratio of pixels/length
    xscale = ax_width / abs(ax_xlim[1] - ax_xlim[0])
    yscale = ax_height / abs(ax_ylim[1] - ax_ylim[0])

    # this is to deal with 'equal' axes aspects
    if bool_auto:

        pass

    else:

        xscale = 1.0
        yscale = 1.0

    # convert length to pixels, 
    # need to minus the lower limit to move the points back to the origin. Then add the limits back on end.
    pt1[0] = (pt1[0] - ax_xlim[0]) * xscale
    pt1[1] = (pt1[1] - ax_ylim[0]) * yscale
    pt2[0] = (pt2[0] - ax_xlim[0]) * xscale
    pt2[1] = (pt2[1] - ax_ylim[0]) * yscale

    # calculate the angle
    theta = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])

    # calculate the radius of the arcs
    r = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]) * k_r

    # arc1 centre
    x11 = pt1[0] + r * np.cos(theta)
    y11 = pt1[1] + r * np.sin(theta)

    # arc2 centre
    x22 = (pt2[0] + pt1[0]) / 2.0 - 2.0 * r * np.sin(theta) - r * np.cos(theta)
    y22 = (pt2[1] + pt1[1]) / 2.0 + 2.0 * r * np.cos(theta) - r * np.sin(theta)

    # arc3 centre
    x33 = (pt2[0] + pt1[0]) / 2.0 - 2.0 * r * np.sin(theta) + r * np.cos(theta)
    y33 = (pt2[1] + pt1[1]) / 2.0 + 2.0 * r * np.cos(theta) + r * np.sin(theta)

    # arc4 centre
    x44 = pt2[0] - r * np.cos(theta)
    y44 = pt2[1] - r * np.sin(theta)

    # prepare the rotated
    q = np.linspace(theta, theta + np.pi/2.0, 50)

    # reverse q
    # t = np.flip(q) # this command is not supported by lower version of numpy
    t = q[::-1]

    # arc coordinates
    arc1x = r * np.cos(t + np.pi/2.0) + x11
    arc1y = r * np.sin(t + np.pi/2.0) + y11

    arc2x = r * np.cos(q - np.pi/2.0) + x22
    arc2y = r * np.sin(q - np.pi/2.0) + y22

    arc3x = r * np.cos(q + np.pi) + x33
    arc3y = r * np.sin(q + np.pi) + y33

    arc4x = r * np.cos(t) + x44
    arc4y = r * np.sin(t) + y44

    # convert back to the axis coordinates
    arc1x = arc1x / xscale + ax_xlim[0]
    arc2x = arc2x / xscale + ax_xlim[0]
    arc3x = arc3x / xscale + ax_xlim[0]
    arc4x = arc4x / xscale + ax_xlim[0]

    arc1y = arc1y / yscale + ax_ylim[0]
    arc2y = arc2y / yscale + ax_ylim[0]
    arc3y = arc3y / yscale + ax_ylim[0]
    arc4y = arc4y / yscale + ax_ylim[0]

    # log scale consideration
    if 'log' in ax.get_xaxis().get_scale():

        for i in range(0, len(arc1x)):

            if arc1x[i] > 0.0:

                arc1x[i] = np.exp(arc1x[i])

            elif arc1x[i] < 0.0:

                arc1x[i] = -np.exp(abs(arc1x[i]))

            else:

                arc1x[i] = 0.0

        for i in range(0, len(arc2x)):

            if arc2x[i] > 0.0:

                arc2x[i] = np.exp(arc2x[i])

            elif arc2x[i] < 0.0:

                arc2x[i] = -np.exp(abs(arc2x[i]))

            else:

                arc2x[i] = 0.0

        for i in range(0, len(arc3x)):

            if arc3x[i] > 0.0:

                arc3x[i] = np.exp(arc3x[i])

            elif arc3x[i] < 0.0:

                arc3x[i] = -np.exp(abs(arc3x[i]))

            else:

                arc3x[i] = 0.0

        for i in range(0, len(arc4x)):

            if arc4x[i] > 0.0:

                arc4x[i] = np.exp(arc4x[i])

            elif arc4x[i] < 0.0:

                arc4x[i] = -np.exp(abs(arc4x[i]))

            else:

                arc4x[i] = 0.0

    else:

        pass

    if 'log' in ax.get_yaxis().get_scale():

        for i in range(0, len(arc1y)):

            if arc1y[i] > 0.0:

                arc1y[i] = np.exp(arc1y[i])

            elif arc1y[i] < 0.0:

                arc1y[i] = -np.exp(abs(arc1y[i]))

            else:

                arc1y[i] = 0.0

        for i in range(0, len(arc2y)):

            if arc2y[i] > 0.0:

                arc2y[i] = np.exp(arc2y[i])

            elif arc2y[i] < 0.0:

                arc2y[i] = -np.exp(abs(arc2y[i]))

            else:

                arc2y[i] = 0.0

        for i in range(0, len(arc3y)):

            if arc3y[i] > 0.0:

                arc3y[i] = np.exp(arc3y[i])

            elif arc3y[i] < 0.0:

                arc3y[i] = -np.exp(abs(arc3y[i]))

            else:

                arc3y[i] = 0.0

        for i in range(0, len(arc4y)):

            if arc4y[i] > 0.0:

                arc4y[i] = np.exp(arc4y[i])

            elif arc4y[i] < 0.0:

                arc4y[i] = -np.exp(abs(arc4y[i]))

            else:

                arc4y[i] = 0.0

    else:

        pass

    # plot arcs
    ax.plot(arc1x, arc1y, **kwargs)
    ax.plot(arc2x, arc2y, **kwargs)
    ax.plot(arc3x, arc3y, **kwargs)
    ax.plot(arc4x, arc4y, **kwargs)

    # plot lines
    ax.plot([arc1x[-1], arc2x[1]], [arc1y[-1], arc2y[1]], **kwargs)
    ax.plot([arc3x[-1], arc4x[1]], [arc3y[-1], arc4y[1]], **kwargs)

    summit = [arc2x[-1], arc2y[-1]]

    if str_text:

        int_line_num = int(int_line_num)

        str_temp = '\n' * int_line_num
        
        # convert radians to degree and within 0 to 360
        ang = np.degrees(theta) % 360.0

        if (ang >= 0.0) and (ang <= 90.0):

            rotation = ang

            str_text = str_text + str_temp

        if (ang > 90.0) and (ang < 270.0):

            rotation = ang + 180.0

            str_text = str_temp + str_text

        elif (ang >= 270.0) and (ang <= 360.0):

            rotation = ang

            str_text = str_text + str_temp

        else:

            rotation = ang

        ax.axes.text(arc2x[-1], arc2y[-1], str_text, ha='center', va='center', rotation=rotation, fontdict=fontdict)

    else:

        pass

    arc1 = [arc1x, arc1y]
    arc2 = [arc2x, arc2y]
    arc3 = [arc3x, arc3y]
    arc4 = [arc4x, arc4y]

    return theta, summit, arc1, arc2, arc3, arc4

