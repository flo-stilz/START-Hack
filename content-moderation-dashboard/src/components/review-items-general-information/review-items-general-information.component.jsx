import React, {useEffect, useState} from 'react';
import './review-items-general-information.styles.scss';
import Spinner from 'react-bootstrap/Spinner'


function ReviewItemsGeneralInformation(props) {

    return (
        <div>
            <p><strong>Player Assessment: </strong>{props.assessment === "" ? (<Spinner animation="border" style={{"marginLeft": "0.5rem"}} size={"sm"}/>) : props.assessment}</p>
            <p><strong>Number of times the reportee was reported:</strong> {props.amountOfTimesReported}</p>
            <p><strong>Game ID:</strong> {props.gameId}</p>
        </div>
    )
}

export default ReviewItemsGeneralInformation;