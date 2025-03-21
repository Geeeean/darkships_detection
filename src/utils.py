import mplcursors


class Utils:
    @staticmethod
    def add_hover_tooltip(scatter_plot, labels):
        """Adds a tooltip on hover for a scatter plot."""
        cursor = mplcursors.cursor(scatter_plot, hover=True)

        @cursor.connect("add")
        def on_hover(sel):
            # Set the text for the tooltip with corresponding label
            sel.annotation.set_text(labels[sel.index])

            # Customize tooltip appearance
            sel.annotation.get_bbox_patch().set_facecolor("white")  # White background
            sel.annotation.get_bbox_patch().set_alpha(
                0.6
            )  # Semi-transparent background

        return cursor
