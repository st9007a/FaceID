import './index.pug'

import 'semantic-ui-offline/semantic.css'
import './app.sass'

import * as tf from '@tensorflow/tfjs'
import { loadFrozenModel } from '@tensorflow/tfjs-converter'

window.$ = window.jQuery = require('jquery')

const MODEL_URL = './model/tensorflowjs_model.pb'
const WEIGHTS_URL = './model/weights_manifest.json'

const video = document.getElementsByTagName('video')[0]
const canvas = document.getElementsByTagName('canvas')[0]

const width = 320
const height = 240

let faceCollection = []
let faceId = null

let isCapture = false
let isValidate = false
let captureProcess = null

video.setAttribute('width', width)
video.setAttribute('height', height)
canvas.setAttribute('width', width)
canvas.setAttribute('height', height)

navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia

loadFrozenModel(MODEL_URL, WEIGHTS_URL).then(main).catch(err => alert(err))

function shapshot() {
  const context = canvas.getContext('2d')
  context.drawImage(video, 0, 0, width, height)
  // tf.fromPixels(context.getImageData(60, 20, 200, 200)).print(1)
  // return tf.fromPixels(context.getImageData(60, 20, 200, 200))
  return context.getImageData(60, 20, 200, 200)
}

function main(model) {
  const successCallback = (stream) => {
    video.srcObject = stream
  }

  const errorCallback = (error) => {
    console.log("navigator.getUserMedia error: ", error)
  }

  navigator.getUserMedia({audio: false, video: true}, successCallback, errorCallback)

  $('#capture').click(e => {

    $(e.currentTarget).toggleClass('teal').toggleClass('red')
    isCapture = !isCapture

    if (isCapture) {
      $(e.currentTarget).text('Stop')
      captureProcess = setInterval(() => faceCollection.push(shapshot()), 500)
    } else {
      clearInterval(captureProcess)

      $(e.currentTarget).text('Build Your Face ID ... ').ready(() => {

        const input = tf.tidy(() => tf.cast(tf.stack(faceCollection.map(el => tf.fromPixels(el))), 'float32'))
        const output = model.execute({ 'squeeze_net/face_input': input })

        faceId = output.dataSync()

        // $(e.currentTarget).text('Capture Your Face ID')
        $(e.currentTarget).text(faceId.length / 128)
        console.log(faceId.length / 128)

        faceCollection.length = 0
      })

    }
  })

  $('#validate').click(e => {
    if (!faceId || faceId.length === 0) {
      alert('You don\'t have face id.')
    }

    if (isCapture) {
      return
    }

    $(e.currentTarget).toggleClass('blue').toggleClass('red')
    isValidate = !isValidate

    if (isValidate) {
      $(e.currentTarget).text('Stop validate').ready(() => {

        captureProcess = setInterval(() => {
          const input = tf.tidy(() => tf.cast(tf.fromPixels(shapshot()).expandDims(), 'float32'))
          const output = model.execute({ 'squeeze_net/face_input': input })

          const target = output.dataSync()

          let dist = 0
          let vote = 0

          for (let i = 0; i < faceId.length; i++) {

            dist += Math.pow(faceId[i] - target[i % 128], 2)

            if (i % 128 === 127) {
              console.log(dist)
              vote += Math.sqrt(dist) < 0.9 ? 1 : 0
              dist = 0
            }

          }
          console.log(vote)

          if (vote >= faceId.length / 128 * 0.75) {
            $('#validate').click()
            $('#lock').fadeOut('slow')
            $('#unlock').fadeIn('slow')
          }
        }, 500)
      })

    } else {
      clearInterval(captureProcess)
      $(e.currentTarget).text('Validate')
    }
  })

  $('#relock').click(e => {
    $('#unlock').fadeOut('slow')
    $('#lock').fadeIn('slow')
  })
}
