import React from 'react';

 const ReviewItemsInformationOnAlliance = (props) => {
     return (
         <div>
             <p><strong>Alliance ID: </strong>{props.alliance.allianceId}</p>
             <p><strong>Moderator: </strong>{props.alliance.moderator}</p>
             <p><strong>Alliance Description: </strong>{props.alliance.allianceDescription}</p>
             <p><input class="form-check-input" disabled={true} checked={props.alliance.isFamilyFriendly}  type="checkbox" style={{"marginRight": "0.5rem"}} /><strong>Family Friendly</strong></p>
         </div>
     )
 }

 export default ReviewItemsInformationOnAlliance;