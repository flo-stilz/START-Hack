import React from "react";

import Button from 'react-bootstrap/Button';

const ReviewItemsPriorityLabel = (props) => {

    const { priority } = props

    switch (true) {
        case (priority === "low"):
            return <Button variant="secondary" size='sm'>Low</Button>
        case (priority === "standard"):
            return <Button variant="primary" size='sm'>Standard</Button>
        case (priority === "medium"):
            return <Button variant="warning" size='sm'>Medium</Button>
        case (priority === "high"):
            return <Button variant="danger" size='sm'>High</Button>
    }
}

export default ReviewItemsPriorityLabel;