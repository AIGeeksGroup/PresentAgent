const { defineConfig } = require('@vue/cli-service')

module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:9297',
        changeOrigin: true
      },
      '/wsapi': {
        target: 'ws://127.0.0.1:9297',
        ws: true,
        changeOrigin: true,
      },
    },
    port: 8088
  },
  css: {
    loaderOptions: {
      postcss: { postcssOptions: { config: false } }
    }
  }
})
