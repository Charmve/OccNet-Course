// Copyright (c) 2019 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/* global document, console */
/* eslint-disable no-console, no-unused-vars, no-undef */
import React, {PureComponent} from 'react';
import {render} from 'react-dom';
import {
  ChakraProvider,
  Box,
  HStack, Select, Tabs, TabList, TabPanels, Tab, TabPanel,
  Text,
  Link,
  VStack,
  Code,
  Grid,
  theme,
} from '@chakra-ui/react';

import {setXVIZConfig, getXVIZConfig} from '@xviz/parser';
import {
  LogViewer,
  PlaybackControl,
  StreamSettingsPanel,
  MeterWidget,
  TrafficLightWidget,
  TurnSignalWidget,
  XVIZPanel,
  VIEW_MODE
} from 'streetscape.gl';
import {Form} from '@streetscape.gl/monochrome';

import {XVIZ_CONFIG, APP_SETTINGS, MAPBOX_TOKEN, MAP_STYLE, XVIZ_STYLE, CAR} from './constants';

setXVIZConfig(XVIZ_CONFIG);

const TIMEFORMAT_SCALE = getXVIZConfig().TIMESTAMP_FORMAT === 'seconds' ? 1000 : 1;

// __IS_STREAMING__ and __IS_LIVE__ are defined in webpack.config.js
const exampleLog = require(__IS_STREAMING__
  ? './log-from-stream'
  : __IS_LIVE__
    ? './log-from-live'
    : './log-from-file');

class Example extends PureComponent {
  state = {
    dataChoice: 'kitti',
    logK: exampleLog.kittiData,
    logN: exampleLog.nuscenesData,
    styleValue:'light',
    mapStyle: MAP_STYLE['light'],
    settings: {
      viewMode: 'PERSPECTIVE',
      showTooltip: false
    }
  };
 
  handleMapChange = (e) => {
    this.setState({
      mapStyle:MAP_STYLE[e.target.value],
      styleValue:e.target.value
    })
  }

  handleChange = (e) => {
    this.setState({
      dataChoice:e.target.value
    })
  };

  componentDidMount() {
    this.state.logK.on('error', console.error).connect();
    this.state.logN.on('error', console.error).connect();
  }

  _onSettingsChange = changedSettings => {
    this.setState({
      settings: {...this.state.settings, ...changedSettings}
    });
  };

  render() {
    const {logK, logN, settings} = this.state;

    return (
      <ChakraProvider theme={theme}>
        <Box w="100vw" h="100vh">
          <HStack w="100vw" h="100vh">
            <Box w="320px" h="100vh" p="2">
              <Tabs>
                  <TabList>
                    <Tab p={'1'}>Dataset View</Tab>
                    <Tab p={'1'}>Settings</Tab>
                  </TabList>
                  <TabPanels w="320px" p="0">
                    <TabPanel p="1">
                      <VStack w="100%">
                        <Box w="100%" h="35px">
                          <Select placeholder='Select Data Source' onChange={this.handleChange.bind(this)}>
                            <option value='kitti' selected>KITTI</option>
                            <option value='nuscenes'>NuScenes</option>
                          </Select>
                        </Box>
                        <Box w="100%">
                          <XVIZPanel log={this.state.dataChoice === 'kitti' ? logK : logN} name="Camera" />
                        </Box>
                        <Box w="100%">
                          <Form
                            data={APP_SETTINGS}
                            values={this.state.settings}
                            onChange={this._onSettingsChange}
                          />
                        </Box>
                        <Box w="100%" h="100%" p="0">
                          <Tabs>
                            <TabList>
                              <Tab p={'1'}>Charts (Metrics)</Tab>
                              <Tab p={'1'}>Streams (Objects)</Tab>
                            </TabList>
                            <TabPanels>
                              <TabPanel>
                                <Box w="100%">
                                  <XVIZPanel log={this.state.dataChoice === 'kitti' ? logK : logN} name="Metrics" />
                                </Box>
                              </TabPanel>
                              <TabPanel maxHeight={'65vh'} overflowY='scroll'>
                                  <StreamSettingsPanel log={this.state.dataChoice === 'kitti' ? logK : logN} />
                              </TabPanel>
                            </TabPanels>
                          </Tabs>
                        </Box>
                      </VStack>
                    </TabPanel>
                    <TabPanel>
                      <VStack w="320px">
                        <Box w="100%" border="1" borderColor={'gray.200'} rounded="md" p={'1'}>
                          <Select placeholder='Select Map Style' onChange={this.handleMapChange.bind(this)}>
                            <option value='light' selected>Light</option>
                            <option value='dark'>Dark</option>
                          </Select>
                        </Box>
                      </VStack>
                    </TabPanel>
                  </TabPanels>
                </Tabs>
            </Box>
            <VStack w='calc(100vw - 320px)' h="100vh">
                <Box h='calc(100vh - 100px)' w="calc(100vw - 320px)" position={'relative'}>
                  <LogViewer
                    log={this.state.dataChoice === 'kitti' ? logK : logN}
                    mapboxApiAccessToken={MAPBOX_TOKEN}
                    mapStyle={this.state.mapStyle}
                    car={CAR}
                    xvizStyles={XVIZ_STYLE}
                    showTooltip={settings.showTooltip}
                    viewMode={VIEW_MODE[settings.viewMode]}
                  />
                  <VStack border={'1px'} borderColor={'gray.200'} p={2} rounded={'lg'} bg="gray.500"
                    top={2} right={12} position={'absolute'} shadow={'lg'} display={'flex'} flexDirection={'column'}>
                    <Box w="100%" display={'flex'} flexDirection={'row'}>
                      <TurnSignalWidget log={this.state.dataChoice === 'kitti' ? logK : logN} streamName="/vehicle/turn_signal" />
                    </Box>
                    <Box w="100%">
                      <TrafficLightWidget log={this.state.dataChoice === 'kitti' ? logK : logN} streamName="/vehicle/traffic_light" />
                    </Box>
                    <Box w="100%">
                      <MeterWidget
                        log={this.state.dataChoice === 'kitti' ? logK : logN}
                        streamName="/vehicle/acceleration"
                        label="Acceleration"
                        min={-4}
                        max={4}
                      />
                    </Box>
                    <Box w="100%">
                      <MeterWidget
                        log={this.state.dataChoice === 'kitti' ? logK : logN}
                        streamName="/vehicle/velocity"
                        label="Speed"
                        getWarning={x => (x > 6 ? 'FAST' : '')}
                        min={0}
                        max={20}
                      />
                    </Box>
                  </VStack>
                </Box>
                <Box h="100px" w="100%" bg="gray.100"> 
                    <PlaybackControl
                      width="100%"
                      log={this.state.dataChoice === 'kitti' ? logK : logN}
                      formatTimestamp={x => new Date(x * TIMEFORMAT_SCALE).toUTCString()}
                    />
                </Box>
            </VStack>
          </HStack>
        </Box>
      </ChakraProvider>
    );
  }
}

render(<Example />, document.getElementById('app'));
