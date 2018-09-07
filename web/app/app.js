import './index.pug'

import 'semantic-ui-offline/semantic.css'

import * as tf from '@tensorflow/tfjs'
import { loadFrozenModel } from '@tensorflow/tfjs-converter'

window.$ = window.jQuery = require('jquery')
require('semantic-ui-offline/semantic.js')

const MODEL_URL = './model/tensorflowjs_model.pb'
const WEIGHTS_URL = './model/weights_manifest.json'

navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia

loadFrozenModel(MODEL_URL, WEIGHTS_URL).then(main)

let ids = []

function main(model) {
  console.log('Get model!')
  navigator.getUserMedia({audio: false, video: true}, successCallback, errorCallback)

  function successCallback(stream) {
    window.stream = stream
    document.getElementsByTagName('h1')[0].textContent = 'success'

    const video = document.getElementsByTagName('video')[0]
    const canvas = document.getElementsByTagName('canvas')[0]

    let width = 640
    let height = 480

    video.srcObject = stream

    video.setAttribute('width', width)
    video.setAttribute('height', height)
    canvas.setAttribute('width', width)
    canvas.setAttribute('height', height)

    $('button').click(() => {
      console.log(canvas)
      const context = canvas.getContext('2d')

      context.drawImage(video, 0, 0, width, height)
      const data = context.getImageData(140, 220, 200, 200)

      console.log(model)
      const res = model.execute({
        'squeeze_net/face_input': tf.fromPixels(data).reshape([1, 200, 200, 3]).asType('float32')
      })
      // res.print()
      // console.log(res.dataSync()[0])
      ids.push(res.dataSync())

      if (ids.length > 1) {
        for (let i = 1; i < ids.length; i++) {
          let sum = 0
          for (let j = 0; j < 128; j++) {
            sum += Math.pow(ids[i][j] - ids[i - 1][j], 2)
          }
          console.log(sum)
        }
      }
    })

  }

  function errorCallback(error){
    console.log("navigator.getUserMedia error: ", error)
    document.getElementsByTagName('h1')[0].textContent = 'fail'
  }
}
