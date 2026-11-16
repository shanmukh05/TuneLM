import original from './figures/music-facts.jpg';
import performances from './figures/performances.jpg';
import recordings from './figures/recordings.jpg';
import aretha from './figures/aretha-franklin.jpg';
import phonograph from './figures/tinfoil-phonograph.jpg';
import indianMusicHistory from './figures/indian-music-history.jpg';
import indianMusicModern from './figures/indian-music-modern.jpg';

const sheets = [
  [original, [
    ['Nina Simone began singing in an Atlantic City bar to pay for classical piano lessons.', 'https://nmaahc.si.edu/nina-simone', '#f4e7da', '#8a4632'],
    ['ABBA won Eurovision for Sweden in 1974 with “Waterloo,” just three years after groups were first allowed to compete.', 'https://www.eurovision.com/eurovision-song-contest/brighton-1974/', '#eee5f0', '#74517b'],
    ['In Trinidad and Tobago, musicians turned discarded oil drums into steelpans that could play melodies.', 'https://www.nalis.gov.tt/resources/tt-content-guide/steelband/', '#e7eddd', '#456353'],
    ['Chuck Berry’s “Johnny B. Goode” is the only rock ’n’ roll song on the golden records aboard both Voyager spacecraft.', 'https://science.nasa.gov/resource/chuck-berry-and-carl-sagan/', '#e6e8ef', '#4d5670'],
  ]],
  [performances, [
    ['Dolly Parton wrote “I Will Always Love You” to say goodbye to Porter Wagoner when she left his show.', 'https://www.dollyparton.com/news/dolly-leaves-porter-wagoner-show', '#f5e6df', '#924c48'],
    ['Reggae grew out of Kingston’s ska and rocksteady scenes. UNESCO added Jamaica’s reggae tradition to its cultural heritage list in 2018.', 'https://ich.unesco.org/en/decisions/13.COM/10.B.18', '#e8edde', '#4b6243'],
    ['At Woodstock in 1969, Ravi Shankar played sitar in the rain during a set that lasted more than 40 minutes.', 'https://www.woodstock.com/lineup/ravi-shankar/', '#e6ebee', '#4c6276'],
    ['The Beatles played their final live performance on the roof of their London headquarters in January 1969.', 'https://www.thebeatles.com/celebrating-beatles-rooftop-performance', '#eee6dd', '#795145'],
  ]],
  [recordings, [
    ['Kyu Sakamoto’s Japanese-language song “Sukiyaki” reached No. 1 on the Billboard Hot 100 in 1963.', 'https://www.worldradiohistory.com/Archive-All-Music/Billboard/60s/1963/Billboard%201963-06-22.pdf', '#e5eaf1', '#465b76'],
    ['João Gilberto’s guitar on the 1958 recording of “Chega de Saudade” helped introduce the sound of bossa nova.', 'https://exposicoesvirtuais.an.gov.br/index.php/galerias/10-exposicoes/98-capitais-da-bossa-nova.html', '#e3ede8', '#41665d'],
    ['You play a theremin without touching it. One hand controls pitch, and the other controls volume.', 'https://api.moogmusic.com/sites/default/files/2018-09/Etherwave_Plus_Manual_Full.pdf', '#ece6ef', '#665176'],
    ['Clara Schumann was still a teenager when she wrote her Piano Concerto in A minor.', 'https://www.loc.gov/static/programs/national-recording-preservation-board/documents/SchumannPianoTrios.pdf', '#f1e8dd', '#805b45'],
  ]],
  [aretha, [
    ['Aretha Franklin and her sisters added the “sock it to me” refrain to her 1967 recording of Otis Redding’s “Respect.”', 'https://blogs.loc.gov/loc/2018/08/aretha-franklin-and-respect/', '#efe3e5', '#78495b'],
  ]],
  [phonograph, [
    ['Thomas Edison tested his 1877 tinfoil phonograph by recording himself reciting “Mary Had a Little Lamb.”', 'https://www.nps.gov/edis/learn/photosmultimedia/eng-extra-2.htm', '#f0e7d8', '#725a3f'],
  ]],
  [indianMusicHistory, [
    ['The Nāṭyaśāstra, an early Indian treatise on performance, groups instruments into strings, winds, drums and solid percussion such as cymbals.', 'https://ebooks.inflibnet.ac.in/engp11/chapter/natya-sastra/', '#f2e6d4', '#825338'],
    ['In the 13th century, Śārṅgadeva’s Saṅgītaratnākara brought earlier music theory and regional traditions together at the Yādava court of Devagiri.', 'https://pustakalaya-org.github.io/vol09_sangita_natya/chapter02_natyashastra.html#29-commentators-and-later-reception', '#e7ebde', '#4e6348'],
    ['Gauhar Jaan made some of India’s earliest commercial gramophone recordings in 1902, signing off with “My name is Gauhar Jaan!”', 'https://artsandculture.google.com/story/pioneering-musicians-women-superstars-of-the-early-gramophone-era-archive-of-indian-music/bQUhgLc4Ow4A8A', '#f0e3df', '#7c4848'],
    ['In 1966, Carnatic singer M. S. Subbulakshmi gave a recital in the United Nations General Assembly Hall in New York.', 'https://digitallibrary.in.one.un.org/TempPdfFiles/2518_1.pdf', '#e5eaf0', '#475f77'],
  ]],
  [indianMusicModern, [
    ['In 2023, Diljit Dosanjh brought Punjabi music to Coachella with a set that included “G.O.A.T.”', 'https://www.youtube.com/watch?v=o0QGw1LZpxM', '#f1e7dc', '#755243'],
    ['“Naatu Naatu,” the Telugu dance number from RRR, won the 2023 Oscar for Best Original Song.', 'https://www.oscars.org/oscars/ceremonies/2023', '#eee7dc', '#795045'],
    ['Shakti’s This Moment, a fusion album with Zakir Hussain on tabla, won Best Global Music Album at the 2024 Grammys.', 'https://www.grammy.com/artists/zakir-hussain/3379/', '#e7e8ee', '#4e5670'],
    ['Chandrika Tandon, Wouter Kellerman and Eru Matsumoto won a 2025 Grammy for Triveni, an album that pairs Vedic chants with flute and cello.', 'https://chandrikatandon.com/triveni/', '#efe5e7', '#704b5e'],
  ]],
];

export const musicFacts = sheets.flatMap(([image, facts]) => {
  const sheet = new URL(image, import.meta.url).href;
  const single = facts.length === 1;
  return facts.map(([copy, source, paper, mark], index) => ({
    copy, source, paper, mark, sheet,
    size: single ? 'cover' : '200% 200%',
    position: single ? 'center' : `${index % 2 * 100}% ${Math.floor(index / 2) * 100}%`,
  }));
});
