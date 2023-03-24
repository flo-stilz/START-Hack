import React, {useEffect, useState} from 'react';

import './review-items-information-on-the-players.styles.scss';

import {
    Chart as ChartJS,
    RadialLinearScale,
    PointElement,
    LineElement,
    Filler,
} from 'chart.js';
import { Radar } from 'react-chartjs-2';

ChartJS.register(
    RadialLinearScale,
    PointElement,
    LineElement,
    Filler,
);

function ReviewItemsInformationOnThePlayer(props) {
    const data = {
        labels: ['Bullying', 'Sexual Content', 'Vulgarity', 'Drug / Alcohol', 'Public Threat', 'Extremism'],
        datasets: props.datasets,
    };

    return (
        <div>
            <p><strong>Recent messaging traits of players involved in the conversation:</strong></p>
            <div className="center" style={{"width": "80%"}}>
                <Radar data={data} />
            </div>
        </div>
    )
}

export default ReviewItemsInformationOnThePlayer;


