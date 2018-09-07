import './index.pug'

import 'semantic-ui-offline/semantic.css'
import './app.sass'

import * as tf from '@tensorflow/tfjs'
import { loadFrozenModel } from '@tensorflow/tfjs-converter'

window.$ = window.jQuery = require('jquery')
require('semantic-ui-offline/semantic.js')

const MODEL_URL = './model/tensorflowjs_model.pb'
const WEIGHTS_URL = './model/weights_manifest.json'

const video = document.getElementsByTagName('video')[0]
const canvas = document.getElementsByTagName('canvas')[0]

const width = 640
const height = 480

let isCapture = false

video.setAttribute('width', width)
video.setAttribute('height', height)
canvas.setAttribute('width', width)
canvas.setAttribute('height', height)

navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia

loadFrozenModel(MODEL_URL, WEIGHTS_URL).then(main)


function main(model) {
  const successCallback = (stream) => {
    video.srcObject = stream
  }

  const errorCallback = (error) => {
    console.log("navigator.getUserMedia error: ", error)
  }

  navigator.getUserMedia({audio: false, video: true}, successCallback, errorCallback)

  $('.ui.button').click(e => {

    $(e.currentTarget).toggleClass('teal').toggleClass('red')
    isCapture = !isCapture

    $(e.currentTarget).text(isCapture ? 'Stop' : 'Capture Your Face ID')


    // const context = canvas.getContext('2d')
    // context.drawImage(video, 0, 0, width, height)
    //
    // const data = context.getImageData(140, 220, 200, 200)
    //
    // const res = model.execute({
    //   'squeeze_net/face_input': tf.fromPixels(data).reshape([1, 200, 200, 3]).asType('float32')
    // })
    //
    // res.dataSync()
  })

}
