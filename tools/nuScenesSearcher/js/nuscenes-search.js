const pageUrl = url_getbase();
var source; 	//source can be either: image-click (which also is used for url parameters), image-upload, text


window.addEventListener('DOMContentLoaded', (e) => {

	//Event Listeners:
	var imageTiles = document.getElementsByClassName('image-tile');
	for (let i = 0; i < imageTiles.length; i++) {
		imageTiles[i].addEventListener('click', image_click);
	}

	document.getElementById('search-form').addEventListener("submit", function (e) {
		e.preventDefault();
		search_go(document.getElementById('search-text').value, 'text', true);
	});

	document.getElementById('page-logo').addEventListener('click',function(e){
		e.preventDefault();
		image_preview_hide();
		search_text_clear();
		url_clear();
		render_initial();
	});

	image_drop(document.querySelector("#search-text"));
	image_drop(document.querySelector(".image-droppable"));
	

	//Url Parameters:
	var searchQuery = url_get_query();
	if (searchQuery) {
		if (searchQuery.includes('sample_')) {
			search_go(searchQuery, 'image-click');
		}
		else {
			search_go(searchQuery, 'text');
		}
	}
	else {
		render_initial();
	}

	//Tooltips:
	var tooltipElements = document.getElementsByClassName('tooltip');
	for (let i = 0; i < imageTiles.length; i++) {
		tooltipElements[i].addEventListener('mousemove', tooltip_show);
	}
	for (let i = 0; i < imageTiles.length; i++) {
		tooltipElements[i].addEventListener('mouseleave', tooltip_hide);
	}

	//Back button, reload page:
	window.onpopstate = function (e) {
		location.reload();
	}

});

function render_initial() {

	var data = [];

	data[0] = { image: "images/examples/example1.jpeg", sampleId: "sample_rk4tw1az6wbgpjl4", externalId: "n015-2018-07-24-11-03-52+0800__CAM_FRONT_LEFT__1532401622854844.jpg" }
	data[1] = { image: "images/examples/example2.jpeg", sampleId: "sample_d5p7dk0xzxsqopag", externalId: "n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151675187558.jpg" }
	data[2] = { image: "images/examples/example3.jpeg", sampleId: "sample_0xgrsr1uqbcnzpvy", externalId: "n015-2018-07-18-11-50-34+0800__CAM_FRONT__1531886158012465.jpg" }
	data[3] = { image: "images/examples/example4.jpeg", sampleId: "sample_5016rnptlpxcw7oc", externalId: "n015-2018-07-27-11-36-48+0800__CAM_BACK__1532662892537525.jpg" }
	data[4] = { image: "images/examples/example5.jpeg", sampleId: "sample_4o7jjjqh1pe8ugv4", externalId: "n015-2018-07-24-11-22-45+0800__CAM_BACK_RIGHT__1532402868177893.jpg" }
	data[5] = { image: "images/examples/example6.jpeg", sampleId: "sample_baraultot3x7lug8", externalId: "n015-2018-08-01-16-32-59+0800__CAM_FRONT_RIGHT__1533112801670339.jpg" }
	data[6] = { image: "images/examples/example7.jpeg", sampleId: "sample_4bfmvfk9wlpoclo1", externalId: "n008-2018-09-18-15-12-01-0400__CAM_FRONT_LEFT__1537298096754799.jpg" }
	data[7] = { image: "images/examples/example8.jpeg", sampleId: "sample_ipfjtd61n6gf6qdn", externalId: "n008-2018-07-27-12-07-38-0400__CAM_FRONT__1532708060612404.jpg" }
	data[8] = { image: "images/examples/example9.jpeg", sampleId: "sample_77g9qip11zf7fhpx", externalId: "n015-2018-07-24-11-13-19+0800__CAM_FRONT__1532402159612460.jpg" }

	for (i = 0; i < data.length; i++) {
		render_tile(i, data[i]);
	}

}

function search_go(searchString, source = null, updateUrl = false, sampleId = null) {

	clear_tiles();
	tooltip_hide();
	title_update(searchString, source);
	error_hide();
	
	if (source == "image-click") {
		url_update(sampleId, source, updateUrl);
	}
	else if(source == "image-upload"){
		url_clear();
	}
	else {
		url_update(searchString, source, updateUrl);
	}

	if (window.innerWidth < 700) {
		document.getElementById('results-pane').scrollIntoView();
	}

	if (searchString.includes('sample_')) {
		var data = { sampleId: searchString }
	}
	else {
		var data = { data: searchString }
	}

	const requestOptions = {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(data)
	};
	fetch('https://www.nyckel.com/v0.9/functions/ianmziyi5mim4xoa/search?sampleCount=10&includeData=true', requestOptions)
		.then(response => response.json())
		.then(response => {

			var imageData = response['searchSamples'];
			
			if (source == "image-click") {
				image_preview_show(imageData[0]['data'], source);
				imageData.shift();					//Remove the first result now that we've used it for the preview:
			}
			else if (source == "image-upload"){
				image_preview_show(searchString, source);
			}
			if (source == "text"){
				image_preview_hide();
			}
						
			for (i = 0; i < 9; i++) {
				render_tile(i, {
					image: imageData[i]['data'],
					link: 'https://www.nuscenes.org/',
					sampleId: imageData[i]['sampleId'],
					externalId: imageData[i]['externalId']
				})
			}

		})
		.catch(function () {

		});

}

function clear_tiles() {
	for (let i = 0; i < 9; i++) {
		document.getElementById('tile-' + String(i)).style.backgroundImage = "none";
	}
}

function render_tile(id, imageData) {
	document.getElementById('tile-image-' + String(id)).style.backgroundImage = "url(" + imageData['image'] + ")";
	document.getElementById('tile-image-' + String(id)).dataset.id = imageData['sampleId'];
	document.getElementById('tile-text-' + String(id)).textContent = imageData['externalId'];

}

function image_click(e) {
	e.preventDefault();

	var sampleId = e.srcElement.getAttribute("data-id");
	search_go(sampleId, 'image-click', true, sampleId);
	
}

function title_update(searchString, source) {
	if (source == "text") {
		document.title = "nuScenesSearcher - " + searchString;
	}
	else{
		document.title = "nuScenesSearcher";
	}
}

function url_getbase() {
	return window.location.href.split('?')[0]
}

function url_update(searchString, source, updateUrl) {
	if (updateUrl) {
		if (searchString) {
			window.history.pushState(Date().toLocaleString(), "nuScenes Searcher - " + searchString, pageUrl + "?search=" + searchString);
		}
		else {
			window.history.pushState(Date().toLocaleString(), "nuScenes Searcher", pageUrl);
		}
	}
}

function url_clear(){
	window.history.pushState(Date().toLocaleString(), "nuScenes Searcher", pageUrl);
}

function url_get_query() {
	const urlParams = new URLSearchParams(window.location.search);
	return urlParams.get('search');
}

function image_upload_click() {

	const file = document.querySelector('input[type=file]').files[0];
	const reader = new FileReader();

	reader.addEventListener("load", function () {
		if (is_file_image(file) == true) {
			search_go(reader.result, 'image-upload');
		}
		else {
			error_show();
		}
	}, false);

	if (file) {
		
		reader.readAsDataURL(file);
	}

}

function image_drop(image_drop_area) {

	var uploaded_image;

	image_drop_area.addEventListener('dragover', (e) => {
		e.stopPropagation();
		e.preventDefault();
		e.dataTransfer.dropEffect = 'copy';
	});

	image_drop_area.addEventListener('drop', (e) => {
		e.stopPropagation();
		e.preventDefault();
		const fileList = event.dataTransfer.files;

		if (is_file_image(fileList[0]) == true) {
			readImage(fileList[0]);
		}
		else {
			error_show();
		}
	});

	readImage = (file) => {
		const reader = new FileReader();
		reader.addEventListener('load', (e) => {
			uploaded_image = e.target.result;
			search_go(uploaded_image, 'image-upload');
		});
		reader.readAsDataURL(file);
	}
}

function image_preview_show(image, source) {
	search_text_clear();
	document.getElementById("ItemPreview").src = image;
	document.querySelector("#current-search-preview").style.display = 'block';
}

function image_preview_hide() {
	document.querySelector("#current-search-preview").style.display = 'none';
}

function search_text_clear(){
	document.querySelector('#search-text').value = "";
}

function error_show() {
	document.querySelector("#current-search-preview").style.display = 'none';
	document.querySelector('.error-message').style.display = 'inline-block';
}

function error_hide() {
	document.querySelector('.error-message').style.display = 'none';
}

function is_file_image(file) {
	const acceptedImageTypes = ['image/gif', 'image/jpeg', 'image/png'];
	return file && acceptedImageTypes.includes(file['type']);
}

function tooltip_show(e) {
	var
		target = this.dataset.tooltiptarget,
		toolTip = document.getElementById(target),
		x = e.clientX,
		y = e.clientY;

	toolTip.style.top = (y + 20) + 'px';
	toolTip.style.left = (x + 20) + 'px';
	toolTip.style.display = 'block';
}

function tooltip_hide() {
	var toolTip = document.querySelector('.tooltip-popup');
	toolTip.style.display = 'none';
}