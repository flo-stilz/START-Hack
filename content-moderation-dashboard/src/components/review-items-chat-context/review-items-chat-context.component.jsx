import React from "react";
import Card from 'react-bootstrap/Card';

const ReviewItemsChatContext = (props) => {
    const { chats, filteredContent } = props;
    const conversations = chats.map((chat) => {
        return (
            <Card>
                <Card.Body style={filteredContent.length !== 0 && filteredContent.some(function(v) { return chat.text.indexOf(v) >= 0; }) ? {"backgroundColor": "#f5c2c7"} : null} >{chat.author.slice(0, 5)}: {chat.text}</Card.Body>
            </Card>
        )
    })

    return (
        <div>
            {conversations}
        </div>
    )
}

export default ReviewItemsChatContext;
