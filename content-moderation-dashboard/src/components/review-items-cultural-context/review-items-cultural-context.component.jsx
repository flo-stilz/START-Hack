import React from 'react';

const ReviewItemsCulturalContext = (props) => {
    if (props.region === "South East Asia") {
        return "The word swastika comes from Sanskrit: स्वस्तिक, romanized: svastika, meaning 'conducive to well-being'. In Hinduism, the right-facing symbol (clockwise) (卐) is called swastika, symbolizing surya ('sun'), prosperity and good luck."
    } else {
        return "N/A"
    }
}

export default ReviewItemsCulturalContext;