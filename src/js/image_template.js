var svgElem = document.getElementById("svg");
console.log(svgElem);

%mousemove%

// Based on: https://stackoverflow.com/a/40475362
function linspace(startValue, stopValue, cardinality) {
  var arr = [];
  var step = (stopValue - startValue) / (cardinality - 1);
  for (var i = 0; i < cardinality; i++) {
    arr.push(startValue + (step * i));
  }
  return arr;
}

function cellFromRange(cellRange, value) {
  var cell = cellRange.length;
  for (var i = 0; i < cellRange.length-1; i++) {
    if (cellRange[i] <= value & cellRange[i+1] > value) {
      cell = i;
      break;
    }
  }
  return cell
}


function getMousePosition(event) {

  let elementBBox = this.getBoundingClientRect()

  let imageLeft = elementBBox.left;
  let imageTop = elementBBox.top;

  let imageWidth = elementBBox.width;
  let imageHeight = elementBBox.height;

  var mouseX = event.clientX-imageLeft;
  var mouseY = event.clientY-imageTop;

  let module_n = parseInt(this.id.substr(this.id.length - 5));
  let heatmap_size = parseInt(this.id.substr(6,5));

  var widthCellRange = linspace(0,imageWidth,heatmap_size);
  var heightCellRange = linspace(0,imageHeight,heatmap_size);

  let widthCell = cellFromRange(widthCellRange, mouseX);
  let heightCell = heightCellRange.length - cellFromRange(heightCellRange, mouseY) - 2;

  let sensor_n = widthCell + 1;
  let actuator_n = heightCell + 2;

  // Based on: https://stackoverflow.com/a/610415
  var body_filename = `${sensor_n}_${actuator_n}_${module_n}`;

  var bodyvideo = document.getElementById("bodyvideo");
  bodyvideo.poster = `../map_elites_body/${body_filename}.png`;

  var bodysource = document.getElementById("bodysource");
  if (bodysource) {
    bodyvideo.removeChild(bodysource);
  }

  var new_bodysource = document.createElement('source');

  new_bodysource.setAttribute('id', "bodysource");
  new_bodysource.setAttribute('src', bodyvideo.poster + ".mp4");
  new_bodysource.setAttribute('type', 'video/mp4');

  bodyvideo.appendChild(new_bodysource);
  bodyvideo.load()

  var fitnesstext = document.getElementById("fitnesstxt");
  fitnesstext.data = bodyvideo.poster + ".html";
}
