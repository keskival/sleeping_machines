"""Save the computing-with-time charts (drawn by make_pdf's functions) as PNGs for REPORT.md."""
import os

import make_pdf as M

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def main():
    for name, fn in (("e26_learning", M.fig_e26), ("e30_restoration", M.fig_e30), ("e32_frontier", M.fig_e32),
                     ("e37_grokking", M.fig_e37), ("depth_theorem", M.fig_depth_theorem), ("e34_depth", M.fig_e34), ("e37_phase", M.fig_phase), ("e41_depth_grok", M.fig_e41)):
        fig = fn()
        if fig is not None:
            fig.savefig(os.path.join(FIG, name + ".png"), dpi=170, bbox_inches="tight", facecolor="white")
            M.plt.close(fig)
            print("wrote", name)


if __name__ == "__main__":
    main()
