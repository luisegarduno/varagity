# Page preview

The evidence-panel page-preview service ([ADR-010](../adr/ADR-010-document-page-preview.md)):
on-demand locate + render behind the two `…/preview/*` routes — markdown
de-decoration, trigram page scoring and pdfium highlight rects, PNG page
rendering, source resolution (containment + content-hash verification), and
the cached LibreOffice PPTX conversion.

::: varagity.preview

::: varagity.preview.normalize

::: varagity.preview.locate

::: varagity.preview.render

::: varagity.preview.source

::: varagity.preview.convert
