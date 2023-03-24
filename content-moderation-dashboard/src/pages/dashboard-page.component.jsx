import React from 'react';
import './dashboard-page.styles.scss';

import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';

import ReviewItems from "../components/review-items/review-items.component";

class DashboardPage extends React.Component {
    constructor() {
        super();
        this.state = {

        }
    }

    render() {
        return (
            <div className='dashboard-container'>
                <h1 style={{"marginBottom": "2rem"}}>Dashboard</h1>
                <Tabs
                    defaultActiveKey="all-entities"
                    id="dashboard-options"
                    className="mb-3"
                >
                    <Tab eventKey="all-entities" title="All Entities">
                        <ReviewItems filter={"all"}/>
                    </Tab>
                    <Tab eventKey="high-entities" title="High Priority">
                        <ReviewItems filter={"high"}/>
                    </Tab>
                    <Tab eventKey="medium-entities" title="Medium Priority">
                        <ReviewItems filter={"medium"}/>
                    </Tab>
                    <Tab eventKey="standard-entities" title="Standard Priority">
                        <ReviewItems filter={"standard"}/>
                    </Tab>
                    <Tab eventKey="low-entities" title="Low Priority">
                        <ReviewItems filter={"low"}/>
                    </Tab>
                </Tabs>
            </div>
        )
    }
}

export default DashboardPage;