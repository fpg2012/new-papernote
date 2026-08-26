// KaTeX 渲染：页面加载/导航后自动渲染数学公式
// 依赖 katex.min.js + contrib/auto-render.min.js（在 mkdocs.yml extra_javascript 中按序加载）
document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
  })
})
