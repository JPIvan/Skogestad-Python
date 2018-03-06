import matplotlib.pyplot as plot

class ClassProperty(object):
    def __init__(self, func):
        self.func = func
    def __get__(self, inst, cls):
        return self.func(cls)

class plot_format:
    """
    TODO Rework as a dictionary child class?
    """
    _valid_kwargs = (
        'fig', 'fig_title', 'ax_title', 'xlabel', 'ylabel',
        'xlim', 'ylim', 'grid', 'legend', 'subplotadj_left',
        'subplotadj_right', 'subplotadj_top', 'subplotadj_bottom',
        'subplotadj_wspace', 'subplotadj_hspace'
    )
    
    def __init__(self, **kwargs):
    
        invalid_kwargs = [k for k in kwargs if k not in self.__class__.valid_kwargs]
        
        if invalid_kwargs:
            raise Exception("\'{}\' is an invalid argument!".format(invalid_kwargs[0]))
                
        for k in self.__class__.valid_kwargs:
            setattr(self, k, None)
                  
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        self.formats = self.getformats()
            
    def __getitem__(self, k):
        return self.formats[k]

    def __setitem__(self, k, v):
        self.formats[k] = v

    def __call__(self):
        return self.formats
    
    @ClassProperty
    def valid_kwargs(self):
        return self._valid_kwargs

    def getformats(self):
        return {k: getattr(self, k) for k in self.__class__.valid_kwargs}

class bode_plot_format(plot_format):
    _valid_kwargs = (
        'fig', 'fig_title', 'ax_title_mag', 'ax_title_phase',
        'xlabel_mag', 'ylabel_mag', 'xlabel_phase', 'ylabel_phase',
        'xlim', 'ylim', 'grid', 'legend', 'subplotadj_left',
        'subplotadj_right', 'subplotadj_top', 'subplotadj_bottom',
        'subplotadj_wspace', 'subplotadj_hspace'
    )
    
    def __init__(self, **kwargs):
        super(bode_plot_format, self).__init__(**kwargs)
    
    def getformats_oneaxis(self):
        formats = []
        for subplot in ("_mag", "_phase"):
            formats.append({ \
                k: getattr(self, k) \
                for k in plot_format.valid_kwargs \
                if k in bode_plot_format.valid_kwargs
            })
            formats[-1]['ax_title'] = self['ax_title'+subplot]
            formats[-1]['xlabel'] = self['xlabel'+subplot]
            formats[-1]['ylabel'] = self['ylabel'+subplot]
        
        for i, v in enumerate(formats):
            formats[i] = plot_format(**v)
        
        return formats

def plot_setfontsizes(
    figure_title=28, axes_title=24, axes_label=20, legend=16, xtick=16,
    ytick=16, font=8
):
    plot.rc('figure', titlesize=figure_title)
    plot.rc('axes', titlesize=axes_title)
    plot.rc('axes', labelsize=axes_label)
    plot.rc('xtick', labelsize=xtick)
    plot.rc('ytick', labelsize=ytick)
    plot.rc('legend', fontsize=legend)
    plot.rc('font', size=font)

def plot_doformatting(ax1, plotformat, fig=None, ax2 = None):
    """
        Does normal matplotlib formatting stuff.

        Parameters
        ----------
        ax : matplotlib axis to format
        plotformat : plot_format instance

        Returns
        -------
        None
    """
    if ax2 is None:
        if plotformat['fig'] and plotformat['fig_title']:
            plotformat['fig'].suptitle(plotformat['fig_title'])

        if plotformat['ax_title']:
            ax1.set_title(plotformat['ax_title'])

        if plotformat['xlabel']:
            ax1.set_xlabel(plotformat['xlabel'])

        if plotformat['ylabel']:
            ax1.set_ylabel(plotformat['ylabel'])

        if plotformat['xlim']:
            ax1.set_xlim(plotformat['xlim'])

        if plotformat['ylim']:
            ax1.set_ylim(plotformat['ylim'])

        if plotformat['grid']:
            ax1.grid()

        if plotformat['legend']:
            ax1.legend()

        if plotformat['subplotadj_left'] or plotformat['subplotadj_right'] \
            or plotformat['subplotadj_bottom'] or plotformat['subplotadj_top'] \
            or plotformat['subplotadj_wspace'] or plotformat['subplotadj_hspace']:
            plotformat['fig'].subplots_adjust(
                left=plotformat['subplotadj_left'],
                right=plotformat['subplotadj_right'],
                bottom=plotformat['subplotadj_bottom'],
                top=plotformat['subplotadj_top'],
                wspace=plotformat['subplotadj_wspace'],
                hspace=plotformat['subplotadj_hspace']
            )
    else:
        if isinstance(plotformat, bode_plot_format):
            f1, f2 = plotformat.getformats_oneaxis()
            for ax, f in ((ax1, f1), (ax2, f2)):
                f['fig'] = fig
                plot_doformatting(ax, f)
