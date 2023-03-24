import {useEffect, useState} from 'react';

import axios from "axios";

import Table from 'react-bootstrap/Table';
import Dropdown from 'react-bootstrap/Dropdown';
import DropdownButton from 'react-bootstrap/DropdownButton';
import Accordion from 'react-bootstrap/Accordion';
import Button from 'react-bootstrap/Button';
import { Modal } from "react-bootstrap";
import Form from 'react-bootstrap/Form';

import ReviewItemsGeneralInformation from '../review-items-general-information/review-items-general-information.component';
import ReviewItemsPriorityLabel from "../review-items-priority-label/review-items-priority-label.component";
import ReviewItemsInformationOnThePlayer from '../review-items-information-on-the-players/review-items-informations-on-the-players.component';
import ReviewItemsInformationOnAlliance from '../review-items-information-on-alliance/review-items-information-on-alliance.component';

import './review-items.styles.scss';
import ReviewItemsCulturalContext from "../review-items-cultural-context/review-items-cultural-context.component";
import ReviewItemsChatContext from "../review-items-chat-context/review-items-chat-context.component";
function ReviewItems(props) {
    const [selectedRow, setSelectedRow] = useState(0);
    const [filterByViolationType, setFilterByViolationType] = useState("All");
    const [sortByAssessedRisk, setSortByAssessedRisk] = useState("Descending");
    const [escalate, setEscalate] = useState(false);
    const [reviewItems, setReviewItems] = useState([
            {
                region: "Europe",
                playerAssessment: "",
                playerId: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d",
                numberOfPastOffences: 2,
                conversation: {
                    texts: [{text: "already using a negative IQ deck",author: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d"},
                            {text: "don't be such a shame",author: "8e7bd1f9bc0f25efd3010629cfe9284d7f88bc308fb4ecebee3a3c6dba3642b8"}
                    ],
                    filteredContent: [],
                    messagingTraits: [
                        {
                            data: [1.3, 0, 0.2, 0, 2.65, 3.74],
                            backgroundColor: "rgba(255, 99, 132, 0.2)",
                            borderColor: "rgba(255, 99, 132, 1)",
                            borderWidth: 1
                        }
                    ],
                },
                alliance: {
                    allianceId: "005459d84c3e64d9f3589a1f4a345bc23e2b7e7dff6ff16f132860fcc379ffdd",
                    allianceDescription: "We’re a clan that’s all about positivity and teamwork. Join us and help us create a supportive and friendly community for all players.",
                    moderator: "2b35d8e299c9eba82c60ee2a6781217e82131e4940f91005158fa95ace683e79",
                    isFamilyFriendly: true
                },
                priority: "standard",
                assessedRisk: 0.0624,
                playerName: "Max Musterman",
                amountOfTimesReported: 1,
                suggestedAction: "Hate",
                gameId: "Clash of Clans",
                preview: "already using a negative IQ deck",
                date: "10.02.2023",
                violationType: [
                    "Hate Speech"
                ]
            },
            {
                region: "Europe",
                playerId: "8e7bd1f9bc0f25efd3010629cfe9284d7f88bc308fb4ecebee3a3c6dba3642b8",
                playerAssessment: "",
                numberOfPastOffences: 1,
                conversation: {
                    texts: [
                        {text: "looser ",author: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d"},
                        {text: "suck it my son",author: "8e7bd1f9bc0f25efd3010629cfe9284d7f88bc308fb4ecebee3a3c6dba3642b8"},
                        {text: "such it",author: "8e7bd1f9bc0f25efd3010629cfe9284d7f88bc308fb4ecebee3a3c6dba3642b8"},
                        {text: "don't be shy",author: "8e7bd1f9bc0f25efd3010629cfe9284d7f88bc308fb4ecebee3a3c6dba3642b8"},
                        {text: "with this level, baby can play this",author: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d"}
                    ],
                    filteredContent: ["suck it"],
                    messagingTraits: [
                        {
                            data: [4.2, 1.3, 2.0, 0.16, 3.21, 1.92],
                            backgroundColor: "rgba(255, 99, 132, 0.2)",
                            borderColor	: "rgba(255, 99, 132, 1)",
                            borderWidth: 1
                        },
                        {
                            data: [0.2, 3.6, 4.0, 1.06, 0, 0],
                            backgroundColor: "rgba(255, 193, 7, 0.2)",
                            borderColor: "rgba(255, 193, 7, 1)",
                            borderWidth: 1
                        }
                    ]
                },
                alliance: {
                    allianceId: "005459d84c3e64d9f3589a1f4a345bc23e2b7e7dff6ff16f132860fcc379ffdd",
                    allianceDescription: "",
                    moderator: "2b35d8e299c9eba82c60ee2a6781217e82131e4940f91005158fa95ace683e79",
                    isFamilyFriendly: false
                },
                priority: "standard",
                assessedRisk: 0.0012,
                playerName: "Max Musterman",
                amountOfTimesReported: 1,
                suggestedAction: "Bullying",
                gameId: "Clash Royale",
                preview: "suck it my son",
                date: "4.04.2021",
                violationType: [
                    "Vulgarity"
                ]
            },
            {
                region: "North America",
                playerId: "4509730df91ecfe30f02e015e256b0050493134882b2700ad3ed274dd1427e11",
                playerAssessment: "",
                numberOfPastOffences: 5,
                conversation: {
                    texts: [
                        {text: "noob",author: "c88da5728a0deaecfab5a3e10f806a1a5b7ef3b7f5e23468d72f83aa49f2f304"},
                        {text: "you behave like an asshole",author: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d"},
                        {text: "congrats",author: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d"},
                        {text: "looks like at all good deck good game",author: "c88da5728a0deaecfab5a3e10f806a1a5b7ef3b7f5e23468d72f83aa49f2f304"},
                        {text: "you can leave now its just a game",author: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d"},
                    ],
                    filteredContent: ["asshole"],
                    messagingTraits: [
                        {
                            data: [0, 4.21, 4.35, 3.22, 5.13, 2.93],
                            backgroundColor: "rgba(255, 99, 132, 0.2)",
                            borderColor: "rgba(255, 99, 132, 1)",
                            borderWidth: 1
                        }
                    ]
                },
                alliance: {
                    allianceId: "01819e40d17fe6ec975a94f14c2a2917eefbddc3664a99bd60005e7edc59d1d0",
                    allianceDescription: "",
                    moderator: "0c13d2c3abdb548598f6d5c4b34bfbacf46a1721fb970c5964283611975365e1",
                    isFamilyFriendly: false
                },
                priority: "high",
                assessedRisk: 0.924,
                playerName: "Max Musterman",
                amountOfTimesReported: 1,
                suggestedAction: "Unintended",
                gameId: "Brawl Stars",
                preview: "noob",
                date: "24.07.2020",
                violationType: [
                    "Vulgarity",
                    "Harassment"
                ]
            },
            {
                region: "South East Asia",
                playerId: "8886d6c3f1d91384c511782b3749cc8649f1f70b5a4bfbeb6dbda1cfd07a49b9",
                playerAssessment: "",
                numberOfPastOffences: 0,
                conversation: {
                    texts: [
                        {text: "wassup",author: "cbe218a603783285c77289378a57431222b3eb4e8332a21bb79d12ec6e5d4a35"},
                        {text: "have you seen the swastika in my profile?",author: "cbe218a603783285c77289378a57431222b3eb4e8332a21bb79d12ec6e5d4a35"},
                        {text: "nah bruh",author: "fd470b7452dd2c91b7cdeabd8ec3aefb565ea803f6efcad86a23997b8f2086dd"},
                        {text: "dammn, thats deep",author: "fd470b7452dd2c91b7cdeabd8ec3aefb565ea803f6efcad86a23997b8f2086dd"},
                        {text: "aight brb, cya",author: "cbe218a603783285c77289378a57431222b3eb4e8332a21bb79d12ec6e5d4a35"},

                    ],
                    filteredContent: ["swastika"],
                    messagingTraits: [
                        {
                            data: [0, 1.1, 0, 0.64, 1.2, 0],
                            backgroundColor: "rgba(255, 99, 132, 0.2)",
                            borderColor: "rgba(255, 99, 132, 1)",
                            borderWidth: 1
                        },
                        {
                            data: [0.9, 0, 1.5, 1.06, 0, 0],
                            backgroundColor: "rgba(255, 193, 7, 0.2)",
                            borderColor: "rgba(255, 193, 7, 1)",
                            borderWidth: 1
                        }
                    ]
                },
                alliance: {
                    allianceId: "01819e40d17fe6ec975a94f14c2a2917eefbddc3664a99bd60005e7edc59d1d0",
                    allianceDescription: "",
                    moderator: "02ccb548e5b32ae394fa3f2aea61b72798971259343fb725ab5917c866e47176",
                    isFamilyFriendly: true
                },
                priority: "standard",
                assessedRisk: 0.0128,
                playerName: "Max Musterman",
                amountOfTimesReported: 4,
                suggestedAction: "Political Identity",
                gameId: "Clash Of Clans",
                preview: "wassup",
                date: "12.11.2019",
                violationType: [
                    "Inappropriate Sharing",
                    "Extremism"
                ]
            },
            {
                region: "North America",
                playerId: "53c2363ac329451912e2c549e5ee7f9a3da8d8ed533ab308fc7ed740d0aa620b",
                playerAssessment: "",
                numberOfPastOffences: 7,
                conversation: {
                    texts: [
                        {text: "My parents found out guys.",author: "29d97c19f6d3ee2e2203fe4a7f5ce01313add813597949ee7aadcb4ea179d74e"},
                        {text: "If i ever go offline and never come back, i ended it.",author: "29d97c19f6d3ee2e2203fe4a7f5ce01313add813597949ee7aadcb4ea179d74e"},
                        {text: "goddamn snowflake white people",author: "86fa17375fd3bac763f09114771ce2b8c0c82868af987c5d84efa0d9260123b3"},
                        {text: "I can't relate",author: "86fa17375fd3bac763f09114771ce2b8c0c82868af987c5d84efa0d9260123b3"},
                        {text: "Nvm\", \"Forget it",author: "29d97c19f6d3ee2e2203fe4a7f5ce01313add813597949ee7aadcb4ea179d74e"},

                    ],
                    filteredContent: [],
                    messagingTraits: [
                        {
                            data: [4.2, 2.13, 3.75, 6.1, 4.37, 3.46],
                            backgroundColor: "rgba(255, 99, 132, 0.2)",
                            borderColor: "rgba(255, 99, 132, 1)",
                            borderWidth: 1
                        },
                        {
                            data: [1.74, 2.56, 4.22, 1.59, 3.06, 2.97],
                            backgroundColor: "rgba(255, 193, 7, 0.2)",
                            borderColor: "rgba(255, 193, 7, 1)",
                            borderWidth: 1
                        }
                    ]
                },
                alliance: {
                    allianceId: "037c1d098638b5caa37af62a1c4423516789fc355bad1e73c501325c6a056c8f",
                    allianceDescription: "We’re a clan of experienced and dedicated players who are always looking to improve.",
                    moderator: "2e34a295662fe6822b470edbc93d438abb36623b4c1ae11b48a700d64020604a",
                    isFamilyFriendly: false
                },
                priority: "high",
                assessedRisk: 0.51,
                playerName: "Max Musterman",
                amountOfTimesReported: 4,
                suggestedAction: "Dangerous Speech",
                gameId: "Clash Royale",
                preview: "My parents found out guys",
                date: "31.03.2022",
                violationType: [
                    "Inappropriate Sharing",
                    "Aggravation"
                ]
            },
            {
                region: "Europe",
                playerAssessment: "",
                numberOfPastOffences: 1,
                conversation: {
                    texts: [
                        {text: "that's the one I want", author: "cbe218a603783285c77289378a57431222b3eb4e8332a21bb79d12ec6e5d4a35"},
                        {text: "mega miner a dawg", author: "cbe218a603783285c77289378a57431222b3eb4e8332a21bb79d12ec6e5d4a35"},
                        {text: "damn everyone getting him now huh", author: "cbe218a603783285c77289378a57431222b3eb4e8332a21bb79d12ec6e5d4a35"},
                        {text: "Yuh", author: "fd470b7452dd2c91b7cdeabd8ec3aefb565ea803f6efcad86a23997b8f2086dd"},
                        {text: "New emote", author: "fd470b7452dd2c91b7cdeabd8ec3aefb565ea803f6efcad86a23997b8f2086dd"},
                        {text: "I’m better", author: "fd470b7452dd2c91b7cdeabd8ec3aefb565ea803f6efcad86a23997b8f2086dd"},
                        {text: "Nobody has more kisses than me", author: "fd470b7452dd2c91b7cdeabd8ec3aefb565ea803f6efcad86a23997b8f2086dd"},
                        {text: "You’re probably better than me", author: "fd470b7452dd2c91b7cdeabd8ec3aefb565ea803f6efcad86a23997b8f2086dd"},
                        {text: "But2", author: "fd470b7452dd2c91b7cdeabd8ec3aefb565ea803f6efcad86a23997b8f2086dd"},
                        {text: "Nah you’re definitely better", author: "29d97c19f6d3ee2e2203fe4a7f5ce01313add813597949ee7aadcb4ea179d74e"},
                        {text: "I used my stupid sparky challenge deck", author: "fd470b7452dd2c91b7cdeabd8ec3aefb565ea803f6efcad86a23997b8f2086dd"},
                        {text: "Lol", author: "29d97c19f6d3ee2e2203fe4a7f5ce01313add813597949ee7aadcb4ea179d74e"},
                        {text: "E-giant and Ebarbs. That’s nastyyyy lol GG", author: "29d97c19f6d3ee2e2203fe4a7f5ce01313add813597949ee7aadcb4ea179d74e"},
                        {text: "Lol", author: "29d97c19f6d3ee2e2203fe4a7f5ce01313add813597949ee7aadcb4ea179d74e"},
                        {text: "I know I’m a bitch", author: "fd470b7452dd2c91b7cdeabd8ec3aefb565ea803f6efcad86a23997b8f2086dd"},
                        {text: "But it works", author: "fd470b7452dd2c91b7cdeabd8ec3aefb565ea803f6efcad86a23997b8f2086dd"},
                        {text: "Nah you’re cool lol. I’ve got a lot of experience playing them in older ladder so I’ve been there lol.", author: "29d97c19f6d3ee2e2203fe4a7f5ce01313add813597949ee7aadcb4ea179d74e"},
                        {text: "Daddy", author: "fd470b7452dd2c91b7cdeabd8ec3aefb565ea803f6efcad86a23997b8f2086dd"},
                        {text: "I mean yeah I hate the mega knight", author: "fd470b7452dd2c91b7cdeabd8ec3aefb565ea803f6efcad86a23997b8f2086dd"},
                        {text: "Autocorrect is crazy", author: "fd470b7452dd2c91b7cdeabd8ec3aefb565ea803f6efcad86a23997b8f2086dd"},
                        {text: "Shitttttt don’t threaten me with a good time lol", author: "fd470b7452dd2c91b7cdeabd8ec3aefb565ea803f6efcad86a23997b8f2086dd"},
                        {text: "That’s what I’m gonna do to your feet", author: "29d97c19f6d3ee2e2203fe4a7f5ce01313add813597949ee7aadcb4ea179d74e"},
                        {text: "Say lessssss", author: "fd470b7452dd2c91b7cdeabd8ec3aefb565ea803f6efcad86a23997b8f2086dd"},
                        {text: "Jstar stop dick riding the leader you fag", author: "29d97c19f6d3ee2e2203fe4a7f5ce01313add813597949ee7aadcb4ea179d74e"},
                        {text: "Jstar a hotdog rider", author: "86fa17375fd3bac763f09114771ce2b8c0c82868af987c5d84efa0d9260123b3"},
                        {text: "Meat grinder", author: "86fa17375fd3bac763f09114771ce2b8c0c82868af987c5d84efa0d9260123b3"},
                        {text: "Stop sucking up", author: "86fa17375fd3bac763f09114771ce2b8c0c82868af987c5d84efa0d9260123b3"},
                        {text: "Tbh i didn’t know he was leader 😂😂", author: "29d97c19f6d3ee2e2203fe4a7f5ce01313add813597949ee7aadcb4ea179d74e"}],
                    filteredContent: ['dick', 'riding', 'fag'],
                    messagingTraits: [
                        {
                            data: [1.2, 1.3, 1.0, 0.16, 0.21, 0.00],
                            backgroundColor: "rgba(255, 99, 132, 0.2)",
                            borderColor	: "rgba(255, 99, 132, 1)",
                            borderWidth: 1
                        },
                        {
                            data: [0.2, 3.6, 4.0, 1.06, 0, 0],
                            backgroundColor: "rgba(255, 193, 7, 0.2)",
                            borderColor: "rgba(255, 193, 7, 1)",
                            borderWidth: 1
                        }
                    ]
                },
                alliance: {
                    allianceId: "7efc20b4d236a41f1b1e5a5fa70c8cd50ca450a0970aec64fc81937f01887024",
                    allianceDescription: "",
                    moderator: "7efc20b4d236a41f1b1e5a5fa70c8cd50ca450a0970aec64fc81937f01887024",
                    isFamilyFriendly: true
                },
                priority: "medium",
                assessedRisk: 0.2664,
                playerName: "Max Musterman",
                amountOfTimesReported: 1,
                suggestedAction: "Bullying",
                gameId: "Clash Royale",
                preview: "Jstar stop dick riding the leader you fag",
                date: "4.04.2021",
                violationType: [
                    "Bullying"
                ]
            },
            {
                region: "Europe",
                playerAssessment: "",
                numberOfPastOffences: 1,
                conversation: {
                    texts: [
                        {
                            text: "Ummm it’s a shit deck ofc you can’t make it work",
                            author: "d7beb040dedf49509054d0b6b10663ab378c3b1cbf24bc25703c7567d8a1ffa1"
                        },
                        {text: "Kys yes I can", author: "d301995a1d78b2449f5643b68aeebb2b170becde99c6cec472d497a29f756bb6"},
                        {
                            text: "get cooked shut up you use classic logbait 😂you’re just mad bc you can’t make an original deck",
                            author: "411a057c9c3b94f45c18df08a5746de6dd3b48aadeb54ecf745d9e507921e201"
                        },
                        {
                            text: "It works dosent it why do I need original deck if I win with this one",
                            author: "411a057c9c3b94f45c18df08a5746de6dd3b48aadeb54ecf745d9e507921e201"
                        },
                        {
                            text: "Brooo don’t be mean to me",
                            author: "d7beb040dedf49509054d0b6b10663ab378c3b1cbf24bc25703c7567d8a1ffa1"
                        },
                        {
                            text: "using a deck just because it works is honestly so bland and it’s people like you that make the game boring 💀",
                            author: "411a057c9c3b94f45c18df08a5746de6dd3b48aadeb54ecf745d9e507921e201"
                        },
                        {
                            text: "I have autism and just want to have fun",
                            author: "411a057c9c3b94f45c18df08a5746de6dd3b48aadeb54ecf745d9e507921e201"
                        },
                        {
                            text: "oh cry about it",
                            author: "411a057c9c3b94f45c18df08a5746de6dd3b48aadeb54ecf745d9e507921e201"
                        },
                        {
                            text: "Yoooo that’s sooo fucked up you making fun of someone with autos",
                            author: "d301995a1d78b2449f5643b68aeebb2b170becde99c6cec472d497a29f756bb6"
                        },
                        {
                            text: "haha he has autism what a loser",
                            author: "b1db250f30e7d3dca445a79e5f46816d8f61b6a3a9f7d08aabdf1139cd759e89"
                        },
                        {
                            text: "skill issue tbh",
                            author: "b8e054bd657bce29f2535a0125909af5803eb04d7cffcb6aba2dedc3c923a372"
                        },
                    ],
                    filteredContent: [],
                    messagingTraits: [
                        {
                            data: [3.2, 3.3, 1.0, 0.16, 2.21, 0.00],
                            backgroundColor: "rgba(255, 99, 132, 0.2)",
                            borderColor: "rgba(255, 99, 132, 1)",
                            borderWidth: 1
                        },
                        {
                            data: [2.2, 3.6, 0.0, 2.06, 0, 0],
                            backgroundColor: "rgba(255, 193, 7, 0.2)",
                            borderColor: "rgba(255, 193, 7, 1)",
                            borderWidth: 1
                        }
                    ]
                },
                alliance: {
                        allianceId: "7f5e5ccbc6c85cdf9412e0e139615c3f6e96a34bcfb484d916107ce8a185576e",
                        allianceDescription: "",
                        moderator: "e6bacfba411f998217b54e99cb322a4d2d7d72a5db737520cf8da10ce879d598",
                        isFamilyFriendly: false,
                },
                priority: "high",
                    assessedRisk: 0.2784,
                    playerName: "Max Musterman",
                    amountOfTimesReported: 2,
                    suggestedAction: "Bullying",
                    gameId: "Clash Royale",
                    preview: "Clown on them autisms",
                    date: "4.04.2021",
                    violationType: [
                    "Bullying"
                ]
            },
            {
                region: "North America",
                playerAssessment: "",
                playerId: "a0c1bbfff3807146f6917a50571fb3874c4e81f8bf7f4c4c20e81962f2ad46f6",
                numberOfPastOffences: 1,
                conversation: {
                texts: [
                    {text: "4 rotting child corpses or double it and give it to the next person", author: "a0c1bbfff3807146f6917a50571fb3874c4e81f8bf7f4c4c20e81962f2ad46f6"},
                    {text: "double", author: "f1b95aea26c29b907c855f72e8efb18ae468be4014a98580277e57eb54a182d1"},
                    {text: "double", author: "90f9eda7b62e97771b04b8f1c4cfbbded34428fc42cd1af430f5f9a1d3c3fb3b"},
                ],
                filteredContent: [],
                messagingTraits: [
                    {
                        data: [1.2, 3.3, 0.0, 0.16, 0.21, 0.00],
                        backgroundColor: "rgba(255, 99, 132, 0.2)",
                        borderColor	: "rgba(255, 99, 132, 1)",
                        borderWidth: 1
                    },
                    {
                        data: [0.2, 3.6, 4.0, 1.06, 0, 0],
                        backgroundColor: "rgba(255, 193, 7, 0.2)",
                        borderColor: "rgba(255, 193, 7, 1)",
                        borderWidth: 1
                    }
                ]
            },
            alliance: {
                allianceId: "7f5932a24b33c51de690ebceb60e4ffd179ef188f0bec3239857ee98a40b8a47",
                allianceDescription: "",
                moderator: "f1b95aea26c29b907c855f72e8efb18ae468be4014a98580277e57eb54a182d1",
                isFamilyFriendly: false
            },
            priority: "medium",
            assessedRisk: 0.26,
            playerName: "Max Musterman",
            amountOfTimesReported: 0,
            suggestedAction: "Public Threat",
            gameId: "Clash Royale",
            preview: "4 rotten child corpses",
            date: "4.04.2021",
            violationType: [
                "Bullying"
            ]
        },
        {
            region: "Asia",
            playerAssessment: "",
            playerId: "c88da5728a0deaecfab5a3e10f806a1a5b7ef3b7f5e23468d72f83aa49f2f304",
            numberOfPastOffences: 3,
            conversation: {
            texts: [
                {text: "Ok then pu$$y", author: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d"},
                {text: "I was playing for fun and u lost 😂😂😂😂", author: "c88da5728a0deaecfab5a3e10f806a1a5b7ef3b7f5e23468d72f83aa49f2f304"},
                {text: "Yo afraid to go for round 2", author: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d"},
                {text: "I wasn’t trying you dumb fuck😭", author: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d"},
                {text: "Nah u afraid to talk to a women", author: "c88da5728a0deaecfab5a3e10f806a1a5b7ef3b7f5e23468d72f83aa49f2f304"},
                {text: "Wasn’t trying 🤓🤓🤓", author: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d"},
                {text: "I banged your mom I wish I used a condom", author: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d"},
                {text: "Only part that sucked was I had to wait in line first😢", author: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d"},
                {text: "I wish I never met with your mother", author: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d"},
                {text: "A disgrace child", author: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d"},
                {text: "That ugly twat gave me 10 sti’s", author: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d"},
                {text: "My kcok itches", author: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d"},
                {text: "I don’t even know if I am your father because your mom has many", author: "c88da5728a0deaecfab5a3e10f806a1a5b7ef3b7f5e23468d72f83aa49f2f304"}
            ],
            filteredContent: ['you', 'dumb', 'fuck😭','banged', 'your', 'mom', 'condom','My', 'kcok', 'itches'],
            messagingTraits: [
                {
                    data: [1.2, 3.3, 0.0, 0.16, 0.21, 0.00],
                    backgroundColor: "rgba(255, 99, 132, 0.2)",
                    borderColor	: "rgba(255, 99, 132, 1)",
                    borderWidth: 1
                },
                {
                    data: [0.2, 3.6, 4.0, 1.06, 0, 0],
                    backgroundColor: "rgba(255, 193, 7, 0.2)",
                    borderColor: "rgba(255, 193, 7, 1)",
                    borderWidth: 1
                }
            ]
        },
        alliance: {
            allianceId: "7ffea937104820333bfd8345580c4741e32e48e236f71d973bca5ed8ea2e8209",
            allianceDescription: "",
            moderator: "ad1acab0b33470e685f43f60e2b864956a561c1210b3ea5c656766d320bb176d",
            isFamilyFriendly: false
        },
        priority: "medium",
        assessedRisk: 0.168,
        playerName: "Max Musterman",
        amountOfTimesReported: 0,
        suggestedAction: "Public Threat",
        gameId: "Clash Royale",
        preview: "I banged your mom I wish I used a condom",
        date: "4.04.2021",
        violationType: [
            "Bullying"
        ]
        }
    ])

    const [assessment, setAssessment] = useState("");

    useEffect(() => {
        if (props.filter !== "all") {
            let result = reviewItems.filter((item) => {
                return item.priority.includes(props.filter);
            });
            setReviewItems(result);
        }

        // console.log("--------------------")
        // console.log(selectedRow)
        // console.log("-----------------")
        // axios.get("http://127.0.0.1:5000/information?" + result[selectedRow].playerId).then((data) => {
        //     setAssessment(data.data)
        // })
    }, [])

    const [show, setShow] = useState(false);

    const handleClose = () => {
        setShow(false);
    }
    const handleShow = () => setShow(true);

    const handleSubmit = () => {
        reviewItems.splice(selectedRow, 1);
    }

    const handleEscalate = () => {
        setEscalate(!escalate);
    }

    const setSuggestedAction = (index, action) => {
        let updatedReviewItems = [...reviewItems];
        let item = {...updatedReviewItems[index]};
        item.suggestedAction = action;
        updatedReviewItems[index] = item;
        setReviewItems(updatedReviewItems);
    }

    let result = reviewItems;

    if (filterByViolationType !== "All") {
        result = result.filter((item) => {
            return item.violationType.includes(filterByViolationType);
        });
    }

    result = result.sort((a, b) => {
        let sorted;
        if (sortByAssessedRisk === "Ascending") {
            sorted = a.assessedRisk - b.assessedRisk;
        } else {
            sorted = b.assessedRisk - a.assessedRisk;
        }
        return sorted;
    });

    const handleOnTableClick = (index) => {
        setSelectedRow(index);
        console.log("--------------------")
        console.log(index)
        console.log("-----------------")
        axios.get("http://127.0.0.1:5000/information?" + reviewItems[index].playerId).then((data) => {
            setAssessment(data.data)
        });
        setAssessment("")
    }

    result = result.map((item, index) => {
        return (
            <tr style={selectedRow === index ? {backgroundColor: "rgb(236, 236, 236)"} : {}} onClick={() => handleOnTableClick(index)} key={index}>
                <td>
                    <ReviewItemsPriorityLabel priority={item.priority}/>
                </td>
                <td>{item.assessedRisk}</td>
                <td>{item.preview}</td>
                <td>{item.date}</td>
                <td>{item.violationType.join(", ")}</td>
            </tr>
        )
    });

    return (
        <div className='review-items-container'>
            <div className="review-items-table overflow-auto">
                <div style={{"display": "block"}} style={{"marginBottom": "1rem"}}>
                    <p style={{"display": "inline-block", "paddingRight": "0.5rem"}}>Violation Type: </p>
                    <DropdownButton style={{"display": "inline-block", "paddingRight": "1rem"}} size="sm" variant="secondary" id="" title={filterByViolationType}>
                        <Dropdown.Item onClick={() => setFilterByViolationType("All")}>All</Dropdown.Item>
                        <Dropdown.Item onClick={() => setFilterByViolationType("Bullying")}>Bullying</Dropdown.Item>
                        <Dropdown.Item onClick={() => setFilterByViolationType("Sexual Content")}>Sexual Content</Dropdown.Item>
                        <Dropdown.Item onClick={() => setFilterByViolationType("Vulgarity")}>Vulgarity</Dropdown.Item>
                        <Dropdown.Item onClick={() => setFilterByViolationType("Drugs / Alcohol")}>Drugs / Alcohol</Dropdown.Item>
                        <Dropdown.Item onClick={() => setFilterByViolationType("Fraud")}>Fraud</Dropdown.Item>
                        <Dropdown.Item onClick={() => setFilterByViolationType("Public Threat")}>Public Threat</Dropdown.Item>
                        <Dropdown.Item onClick={() => setFilterByViolationType("Extremism")}>Extremism</Dropdown.Item>
                    </DropdownButton>

                    <p style={{"display": "inline-block", "paddingRight": "0.5rem"}}>Assessed Risk: </p>
                    <DropdownButton style={{"display": "inline-block"}} size="sm" variant="secondary" id="" title={sortByAssessedRisk}>
                        <Dropdown.Item onClick={() => setSortByAssessedRisk("Ascending")}>Ascending</Dropdown.Item>
                        <Dropdown.Item onClick={() => setSortByAssessedRisk("Descending")}>Descending</Dropdown.Item>
                    </DropdownButton>
                </div>
                <Table bordered hover>
                    <thead>
                    <tr>
                        <th>Priority</th>
                        <th>Risk</th>
                        <th>Preview</th>
                        <th>Date</th>
                        <th>Violation</th>
                    </tr>
                    </thead>
                    <tbody>
                        {result}
                    </tbody>
                </Table>
            </div>
            {reviewItems.length !== 0 ? (
            <div className='review-items-details overflow-auto'>
                <h4 style={{"marginBottom": "1.5rem"}}>Details</h4>
                <p style={{"display": "inline-block", "paddingRight": "1rem"}}><strong>Date: </strong>{reviewItems[selectedRow].date}</p>
                <ReviewItemsPriorityLabel style={{"display": "inline-block"}} priority={reviewItems[selectedRow].priority}/>

                <p><strong>Player concerned: </strong>{reviewItems[selectedRow].playerName}'s Account</p>
                <p><strong>For review: </strong>{reviewItems[selectedRow].preview}</p>
                <p><strong>Violation type: </strong> {reviewItems[selectedRow].violationType.join(", ")}</p>

                <Accordion alwaysOpen>
                    <Accordion.Item eventKey="0">
                        <Accordion.Header>General Information</Accordion.Header>
                        <Accordion.Body>
                            <ReviewItemsGeneralInformation assessment={assessment} amountOfTimesReported={reviewItems[selectedRow].amountOfTimesReported} gameId={reviewItems[selectedRow].gameId} playerId={reviewItems[selectedRow].playerId}/>
                        </Accordion.Body>
                    </Accordion.Item>
                    <Accordion.Item eventKey="1">
                        <Accordion.Header>Chat Context</Accordion.Header>
                        <Accordion.Body>
                           <ReviewItemsChatContext chats={reviewItems[selectedRow].conversation.texts} filteredContent={reviewItems[selectedRow].conversation.filteredContent}/>
                        </Accordion.Body>
                    </Accordion.Item>
                    <Accordion.Item eventKey="2">
                        <Accordion.Header>Player Interactions</Accordion.Header>
                        <Accordion.Body>
                            <ReviewItemsInformationOnThePlayer datasets={reviewItems[selectedRow].conversation.messagingTraits}/>
                        </Accordion.Body>
                    </Accordion.Item>
                    <Accordion.Item eventKey="3">
                        <Accordion.Header>Information on the Alliance</Accordion.Header>
                        <Accordion.Body>
                            <ReviewItemsInformationOnAlliance
                                alliance={reviewItems[selectedRow].alliance}
                            />
                        </Accordion.Body>
                    </Accordion.Item>
                    <Accordion.Item eventKey="4">
                        <Accordion.Header>Cultural Context</Accordion.Header>
                        <Accordion.Body>
                            <ReviewItemsCulturalContext region={reviewItems[selectedRow].region} />
                        </Accordion.Body>
                    </Accordion.Item>
                </Accordion>
                <Form.Check
                    type="switch"
                    onClick={handleEscalate}
                    style={{"marginTop": "1rem"}}
                    id="custom-switch"
                    label="Escalate to supervisor"
                />
                <Form.Control hidden={!escalate} as="textarea" rows={3} style={{"marginTop": "0.5rem"}} placeholder={"Reason to escalate to supervisor."} />

                <Dropdown style={{"marginTop": "1rem", "display": "inline-block", "marginRight": "0.5rem"}}>
                    <Dropdown.Toggle variant="dark" id="dropdown-basic">
                        {reviewItems[selectedRow].suggestedAction}
                    </Dropdown.Toggle>

                    <Dropdown.Menu>
                        <Dropdown.Item onClick={() => setSuggestedAction(selectedRow, "Spamming / Scamming")}>Spamming / Scamming</Dropdown.Item>
                        <Dropdown.Item onClick={() => setSuggestedAction(selectedRow, "Political Identity")}>Political Identity</Dropdown.Item>
                        <Dropdown.Item onClick={() => setSuggestedAction(selectedRow, "Unintended")}>Unintended</Dropdown.Item>
                        <Dropdown.Item onClick={() => setSuggestedAction(selectedRow, "Pll")}>Pll</Dropdown.Item>
                        <Dropdown.Item onClick={() => setSuggestedAction(selectedRow, "Aggravation")}>Aggravation</Dropdown.Item>
                        <Dropdown.Item onClick={() => setSuggestedAction(selectedRow, "Habuku")}>Habuku</Dropdown.Item>
                        <Dropdown.Item onClick={() => setSuggestedAction(selectedRow, "Abuse of Play")}>Abuse of Play</Dropdown.Item>
                        <Dropdown.Item onClick={() => setSuggestedAction(selectedRow, "Harassment")}>Harassment</Dropdown.Item>
                        <Dropdown.Item onClick={() => setSuggestedAction(selectedRow, "Hate")}>Hate</Dropdown.Item>
                        <Dropdown.Item onClick={() => setSuggestedAction(selectedRow, "Inappropriate sharing")}>Inappropriate sharing</Dropdown.Item>
                        <Dropdown.Item onClick={() => setSuggestedAction(selectedRow, "Dangerous Speech")}>Dangerous Speech</Dropdown.Item>
                        <Dropdown.Item onClick={() => setSuggestedAction(selectedRow, "Extremism")}>Extremism</Dropdown.Item>
                        <Dropdown.Item onClick={() => setSuggestedAction(selectedRow, "Criminal or Predatory")}>Criminal or Predatory</Dropdown.Item>
                        <Dropdown.Divider />
                        <Dropdown.Item onClick={() => setSuggestedAction(selectedRow, "Ignore")}>Ignore</Dropdown.Item>
                    </Dropdown.Menu>
                </Dropdown>
                <Button style={{"display": "inline-block"}} variant="dark" onClick={handleShow}>Submit</Button>
            </div> ) : null }

            {reviewItems.length !== 0 ? (
            <Modal show={show} onHide={handleClose}>
                <Modal.Header closeButton>
                    <Modal.Title>Confirm Changes</Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    {reviewItems[selectedRow].suggestedAction === "Ignore" ? "The reviewed item will be ignored." : "The reviewed item will be classified as " + reviewItems[selectedRow].suggestedAction + "."}
                </Modal.Body>
                <Modal.Footer>
                    <Button variant="secondary" onClick={handleClose}>
                        Close
                    </Button>
                    <Button variant="dark" onClick={() => {
                        handleClose(); handleSubmit();
                    }}>
                        Submit Changes
                    </Button>
                </Modal.Footer>
            </Modal> ) : null}
        </div>
    )
}

export default ReviewItems;
