import autoprefixer from 'autoprefixer'
import { resolve } from 'path'

const config = {
  mode: 'development',
  context: resolve('app'),
  entry: {
    app: './app.js',
  },
  module: {
    rules: [
      {
        test: /\.pug$/,
        use: [
          { loader: 'file-loader', options: { name: '[name].html' } },
          { loader: 'extract-loader' },
          { loader: 'html-loader' },
          { loader: 'pug-html-loader' },
        ],
      },
      {
        test: /\.sass$/,
        use: [
          { loader: 'file-loader', options: { name: '[name].css' } },
          { loader: 'extract-loader' },
          { loader: 'css-loader' },
          { loader: 'postcss-loader', options: { plugins: [autoprefixer] } },
          { loader: 'sass-loader' },
        ],
      },
      {
        test: /\.js$/,
        use: 'babel-loader',
        exclude: resolve('node_modules'),
      },
      {
        test: /\.(eot|ico|jpg|mp3|svg|ttf|woff2|woff|png?)($|\?)/,
        use: 'file-loader',
      },
      {
        test: /\.css$/,
        use: [
          { loader: 'style-loader', options: { insertAt: 'top' } },
          { loader: 'css-loader' },
        ],
      },
      {
        test: /\.wasm$/,
        type: 'javascript/auto',
        use: [
          { loader: 'url-loader', options: { name: '[name].wasm', limit: 1 } },
        ],
      },
    ],
  },
  output: {
    filename: 'app.js',
    path: resolve('dist'),
  },
}

export default config
